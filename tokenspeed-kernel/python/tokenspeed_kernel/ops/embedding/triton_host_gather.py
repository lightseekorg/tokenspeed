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

"""Triton UVA gather of uint8 rows from a host-resident table."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_CAPABILITY = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))
_SIGNATURES = frozenset({format_signature(x=dense_tensor_format(torch.uint8))})


@triton.jit
def _host_uint8_row_gather_kernel(
    table_ptr,
    indices_ptr,
    out_ptr,
    n_rows,
    width,
    table_stride,
    out_stride,
    INDEX_INT64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(indices_ptr + row)
    if not INDEX_INT64:
        idx = idx.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    valid = (idx >= 0) & (idx < n_rows)
    host_row = tl.cast(table_ptr + idx * table_stride, tl.pointer_type(tl.uint8))
    values = tl.load(host_row + cols, mask=mask & valid, other=0)
    tl.store(out_ptr + row * out_stride + cols, values, mask=mask)


# Lookups with at least this many rows are sorted by table row first: random
# rows from a large small-page host table miss the GPU TLB on nearly every
# read, and neighbouring rows share page walks once sorted. Each distinct row
# is then read once. Below it, sorting costs more than it saves.
_SORT_MIN_ROWS = 32768
# Rows per program on the sorted path: their host reads are issued together,
# so more of them are in flight while each waits on the host link. Unsorted
# lookups are small and run one row per program.
_ROWS_PER_PROGRAM = 4
# Canonical UE8M0 exponent for an all-zero / invalid row under the 1e-4 amax
# floor: ceil_log2(1e-4 / 448.0) + 127 = 105 (2^-22).
_FLOOR_EXPONENT = 105
# Bound per-launch row span during in-place table canonicalization so a 100+ GB
# host table completes in short kernels without tripping GPU watchdogs.
_CANONICALIZE_CHUNK_ROWS = 4_194_304


# The entry count varies per batch; keep it out of the specialization key.
@triton.jit(do_not_specialize=["n_entries"])
def _host_mxfp8_row_gather_kernel(
    table_ptr,
    ids_ptr,
    dst_ptr,
    head_ptr,
    codes_ptr,
    scales_ptr,
    n_entries,
    n_rows,
    n_columns,
    index_stride_token,
    index_stride_column,
    table_stride,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
    SORTED: tl.constexpr,
    CANONICAL: tl.constexpr,
):
    # Each program handles ROWS lookup entries. When CANONICAL is False, an
    # entry loads a packed [codes | scales] row, rebuilds its FP32 values and
    # quantizes them again with the group-32 UE8M0 rule (1e-4 amax floor), as
    # quantizing the dequantized BF16 row would. When CANONICAL is True, the
    # table was already canonicalized in place once after load, so valid rows
    # copy their bytes directly and invalid ids emit zero codes with the floor
    # exponent. SORTED entries come in table-row order and only the first entry
    # of each run of equal ids (head[e] == e) reads the host; it writes its
    # output row dst[e], which _copy_duplicate_rows_kernel then copies to the
    # rest of the run.
    entry = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = entry < n_entries
    if SORTED:
        idx = tl.load(ids_ptr + entry, mask=live, other=-1).to(tl.int64)
        dst = tl.load(dst_ptr + entry, mask=live, other=0).to(tl.int64)
        live = live & (tl.load(head_ptr + entry, mask=live, other=-1) == entry)
    else:
        token = entry // n_columns
        column = entry % n_columns
        idx = tl.load(
            ids_ptr + token * index_stride_token + column * index_stride_column,
            mask=live,
            other=-1,
        ).to(tl.int64)
        dst = entry
    valid = live & (idx >= 0) & (idx < n_rows)
    host_rows = tl.cast(
        table_ptr + tl.where(valid, idx, 0) * table_stride, tl.pointer_type(tl.uint8)
    )
    cols = tl.arange(0, DIM)
    groups = tl.arange(0, DIM // 32)
    codes = tl.load(host_rows[:, None] + cols[None, :], mask=valid[:, None], other=0)
    if CANONICAL:
        exponents = tl.load(
            host_rows[:, None] + DIM + groups[None, :],
            mask=valid[:, None],
            other=_FLOOR_EXPONENT,
        )
        tl.store(
            codes_ptr + dst[:, None] * DIM + cols[None, :],
            codes,
            mask=live[:, None],
        )
        tl.store(
            scales_ptr + dst[:, None] * (DIM // 32) + groups[None, :],
            exponents,
            mask=live[:, None],
        )
    else:
        exponents = tl.load(
            host_rows[:, None] + DIM + groups[None, :], mask=valid[:, None], other=0
        )
        scale = (exponents.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        values = tl.reshape(
            codes.to(tl.float8e4nv, bitcast=True).to(tl.float32), (ROWS, DIM // 32, 32)
        ) * tl.expand_dims(scale, 2)
        amax = tl.maximum(tl.max(tl.abs(values), 2), 1.0e-4)
        bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        inverse_scale = 1.0 / (exponent << 23).to(tl.float32, bitcast=True)
        quantized = (values * tl.expand_dims(inverse_scale, 2)).to(tl.float8e4nv)
        tl.store(
            codes_ptr + dst[:, None] * DIM + cols[None, :],
            tl.reshape(quantized.to(tl.uint8, bitcast=True), (ROWS, DIM)),
            mask=live[:, None],
        )
        tl.store(
            scales_ptr + dst[:, None] * (DIM // 32) + groups[None, :],
            exponent.to(tl.uint8),
            mask=live[:, None],
        )


# Row bounds vary per table and chunk; keep them out of the compile key.
@triton.jit(do_not_specialize=["start_row", "end_row"])
def _canonicalize_mxfp8_table_kernel(
    table_ptr,
    start_row,
    end_row,
    table_stride,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Canonicalize [codes | scales] rows in place to the group-32 UE8M0
    # activation rule (1e-4 amax floor), matching _host_mxfp8_row_gather_kernel
    # bit for bit so canonical_mxfp8_row_gather needs no runtime requantization.
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS) + start_row
    live = row < end_row
    host_rows = tl.cast(
        table_ptr + tl.where(live, row, 0) * table_stride, tl.pointer_type(tl.uint8)
    )
    cols = tl.arange(0, DIM)
    groups = tl.arange(0, DIM // 32)
    codes = tl.load(host_rows[:, None] + cols[None, :], mask=live[:, None], other=0)
    exponents = tl.load(
        host_rows[:, None] + DIM + groups[None, :], mask=live[:, None], other=0
    )
    scale = (exponents.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    values = tl.reshape(
        codes.to(tl.float8e4nv, bitcast=True).to(tl.float32), (ROWS, DIM // 32, 32)
    ) * tl.expand_dims(scale, 2)
    amax = tl.maximum(tl.max(tl.abs(values), 2), 1.0e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inverse_scale = 1.0 / (exponent << 23).to(tl.float32, bitcast=True)
    quantized = (values * tl.expand_dims(inverse_scale, 2)).to(tl.float8e4nv)
    tl.store(
        host_rows[:, None] + cols[None, :],
        tl.reshape(quantized.to(tl.uint8, bitcast=True), (ROWS, DIM)),
        mask=live[:, None],
    )
    tl.store(
        host_rows[:, None] + DIM + groups[None, :],
        exponent.to(tl.uint8),
        mask=live[:, None],
    )


@triton.jit(do_not_specialize=["n_entries"])
def _copy_duplicate_rows_kernel(
    dst_ptr,
    head_ptr,
    codes_ptr,
    scales_ptr,
    n_entries,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Copy each sorted run's output row, written by its first entry, to the
    # output rows of the run's other entries.
    entry = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = entry < n_entries
    head = tl.load(head_ptr + entry, mask=live, other=0).to(tl.int64)
    duplicate = live & (head != entry)
    src = tl.load(dst_ptr + head, mask=duplicate, other=0).to(tl.int64)
    dst = tl.load(dst_ptr + entry, mask=duplicate, other=0).to(tl.int64)
    cols = tl.arange(0, DIM)
    groups = tl.arange(0, DIM // 32)
    codes = tl.load(
        codes_ptr + src[:, None] * DIM + cols[None, :], mask=duplicate[:, None]
    )
    tl.store(
        codes_ptr + dst[:, None] * DIM + cols[None, :], codes, mask=duplicate[:, None]
    )
    scales = tl.load(
        scales_ptr + src[:, None] * (DIM // 32) + groups[None, :],
        mask=duplicate[:, None],
    )
    tl.store(
        scales_ptr + dst[:, None] * (DIM // 32) + groups[None, :],
        scales,
        mask=duplicate[:, None],
    )


def _launch_mxfp8_row_gather(
    table: torch.Tensor,
    indices: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    canonical: bool,
) -> None:
    dim = table.shape[1] // 33 * 32
    entries = indices.shape[0] * indices.shape[1]
    if entries == 0:
        return
    table_ptr = current_platform().device_visible_data_ptr(table)
    codes_bytes = codes.view(torch.uint8)
    if entries < _SORT_MIN_ROWS:
        _host_mxfp8_row_gather_kernel[(entries,)](
            table_ptr=table_ptr,
            ids_ptr=indices,
            dst_ptr=indices,  # Unused when SORTED=False.
            head_ptr=indices,  # Unused when SORTED=False.
            codes_ptr=codes_bytes,
            scales_ptr=scales,
            n_entries=entries,
            n_rows=table.shape[0],
            n_columns=indices.shape[1],
            index_stride_token=indices.stride(0),
            index_stride_column=indices.stride(1),
            table_stride=table.stride(0),
            DIM=dim,
            ROWS=1,
            SORTED=False,
            CANONICAL=canonical,
        )
        return
    # Everything below stays on the device: no host sync on the number of
    # distinct rows, so the grids do not depend on the ids.
    sorted_ids, dst = torch.sort(indices.reshape(-1))
    # head[e]: position of the first entry of e's run of equal ids.
    # These positions address dense outputs, not host-table rows. Kernels
    # widen them to int64 before multiplying by output strides.
    head = torch.searchsorted(sorted_ids, sorted_ids)
    grid = (triton.cdiv(entries, _ROWS_PER_PROGRAM),)
    _host_mxfp8_row_gather_kernel[grid](
        table_ptr=table_ptr,
        ids_ptr=sorted_ids,
        dst_ptr=dst,
        head_ptr=head,
        codes_ptr=codes_bytes,
        scales_ptr=scales,
        n_entries=entries,
        n_rows=table.shape[0],
        n_columns=1,  # Column count and index strides are unused when sorted.
        index_stride_token=1,
        index_stride_column=0,
        table_stride=table.stride(0),
        DIM=dim,
        ROWS=_ROWS_PER_PROGRAM,
        SORTED=True,
        CANONICAL=canonical,
    )
    _copy_duplicate_rows_kernel[grid](
        dst,
        head,
        codes_bytes,
        scales,
        entries,
        DIM=dim,
        ROWS=_ROWS_PER_PROGRAM,
    )


@register_kernel(
    "embedding",
    "host_mxfp8_row_gather",
    name="triton_host_mxfp8_row_gather",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_host_mxfp8_row_gather(
    table: torch.Tensor,
    indices: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    _launch_mxfp8_row_gather(table, indices, codes, scales, canonical=False)


@register_kernel(
    "embedding",
    "host_canonical_mxfp8_row_gather",
    name="triton_host_canonical_mxfp8_row_gather",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_host_canonical_mxfp8_row_gather(
    table: torch.Tensor,
    indices: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    _launch_mxfp8_row_gather(table, indices, codes, scales, canonical=True)


@register_kernel(
    "embedding",
    "host_canonicalize_mxfp8_table",
    name="triton_host_canonicalize_mxfp8_table",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_host_canonicalize_mxfp8_table(
    table: torch.Tensor,
    start_row: int,
    end_row: int,
    device: torch.device,
) -> None:
    dim = table.shape[1] // 33 * 32
    if start_row >= end_row:
        return
    table_ptr = current_platform().device_visible_data_ptr(table)
    table_stride = table.stride(0)
    with torch.cuda.device(device):
        for chunk_start in range(start_row, end_row, _CANONICALIZE_CHUNK_ROWS):
            chunk_end = min(chunk_start + _CANONICALIZE_CHUNK_ROWS, end_row)
            grid = (triton.cdiv(chunk_end - chunk_start, _ROWS_PER_PROGRAM),)
            _canonicalize_mxfp8_table_kernel[grid](
                table_ptr=table_ptr,
                start_row=chunk_start,
                end_row=chunk_end,
                table_stride=table_stride,
                DIM=dim,
                ROWS=_ROWS_PER_PROGRAM,
            )


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@register_kernel(
    "embedding",
    "host_uint8_row_gather",
    name="triton_host_uint8_row_gather",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_host_uint8_row_gather(
    table: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    flat = indices.reshape(-1)
    width = table.shape[1]
    table_bytes = table.view(torch.uint8)
    table_ptr = current_platform().device_visible_data_ptr(table_bytes)
    grid = (flat.numel(),)
    _host_uint8_row_gather_kernel[grid](
        table_ptr,
        flat,
        out.reshape(-1, width),
        table.shape[0],
        width,
        table_bytes.stride(0),
        width,
        INDEX_INT64=flat.dtype == torch.int64,
        BLOCK=_next_power_of_2(width),
    )
    return out
