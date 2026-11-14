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
):
    # Each program handles ROWS lookup entries. An entry loads a packed
    # [codes | scales] row, rebuilds its FP32 values and quantizes them again
    # with the group-32 UE8M0 rule (1e-4 amax floor), as quantizing the
    # dequantized BF16 row would. SORTED entries come in table-row order and
    # only the first entry of each run of equal ids (head[e] == e) reads the
    # host; it writes its output row dst[e], which
    # _copy_duplicate_rows_kernel then copies to the rest of the run.
    entry = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = entry < n_entries
    if SORTED:
        idx = tl.load(ids_ptr + entry, mask=live, other=-1).to(tl.int64)
        dst = tl.load(dst_ptr + entry, mask=live, other=0)
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
    head = tl.load(head_ptr + entry, mask=live, other=0)
    duplicate = live & (head != entry)
    src = tl.load(dst_ptr + head, mask=duplicate, other=0)
    dst = tl.load(dst_ptr + entry, mask=duplicate, other=0)
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
    dim = table.shape[1] // 33 * 32
    entries = indices.shape[0] * indices.shape[1]
    if entries == 0:
        return
    table_ptr = current_platform().device_visible_data_ptr(table)
    codes_bytes = codes.view(torch.uint8)
    if entries < _SORT_MIN_ROWS:
        _host_mxfp8_row_gather_kernel[(entries,)](
            table_ptr,
            indices,
            indices,
            indices,
            codes_bytes,
            scales,
            entries,
            table.shape[0],
            indices.shape[1],
            indices.stride(0),
            indices.stride(1),
            table.stride(0),
            DIM=dim,
            ROWS=1,
            SORTED=False,
        )
        return
    # Everything below stays on the device: no host sync on the number of
    # distinct rows, so the grids do not depend on the ids.
    sorted_ids, dst = torch.sort(indices.reshape(-1))
    # head[e]: position of the first entry of e's run of equal ids.
    head = torch.searchsorted(sorted_ids, sorted_ids)
    grid = (triton.cdiv(entries, _ROWS_PER_PROGRAM),)
    _host_mxfp8_row_gather_kernel[grid](
        table_ptr,
        sorted_ids,
        dst,
        head,
        codes_bytes,
        scales,
        entries,
        table.shape[0],
        1,
        1,
        0,
        table.stride(0),
        DIM=dim,
        ROWS=_ROWS_PER_PROGRAM,
        SORTED=True,
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
