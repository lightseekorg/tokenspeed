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

"""Expand decode page tables and produce their consumers in one launch."""

from typing import NamedTuple

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


class _GroupBuffers(NamedTuple):
    source: torch.Tensor
    source_stride: tuple[int, int]
    columns: int
    leaf_table: torch.Tensor
    leaf_lengths: torch.Tensor
    leaf_strides: tuple[int, int, int]

    # Triton's field-level contract also covers nested stride tuples and
    # changing source-view alignment; top-level JIT flags do not recurse.
    __triton_do_not_specialize__ = (
        "source",
        "source_stride",
        "columns",
        "leaf_table",
        "leaf_lengths",
        "leaf_strides",
    )


@triton.jit(
    do_not_specialize=[
        "actual_bs",
        "table_stride",
        "group_stride",
        "location_stride",
        "tokens_per_req",
    ],
    do_not_specialize_on_alignment=[
        "actual_bs",
        "table_stride",
        "group_stride",
        "location_stride",
        "tokens_per_req",
    ],
)
def _decode_tables_kernel(
    groups,
    tables,
    table_stride,
    group_stride,
    locations,
    location_stride,
    lengths,
    actual_bs,
    tokens_per_req,
    RATIOS: tl.constexpr,
    ATOM_PAGES: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE_SIZES: tl.constexpr,
    FLOORS: tl.constexpr,
    COPIES: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    tile = tl.program_id(2)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    for g in tl.static_range(len(RATIOS)):
        if group == g:
            group_buffers = groups[g]
            source = group_buffers.source + row * group_buffers.source_stride[0]
            raw_col = col // RATIOS[g]
            raw = tl.load(
                source + raw_col.to(tl.int64) * group_buffers.source_stride[1],
                (row < actual_bs)
                & (raw_col < group_buffers.columns)
                & (col < PAGES[g]),
                other=0,
            ).to(tl.int64)
            page = tl.where(raw > 0, raw * ATOM_PAGES[g] + col % RATIOS[g], 0)
            tl.store(
                tables + g * group_stride.to(tl.int64) + row * table_stride + col,
                page,
                col < CAPACITY,
            )
            if COPIES[g]:
                tl.store(
                    group_buffers.leaf_table
                    + row * group_buffers.leaf_strides[0]
                    + col.to(tl.int64) * group_buffers.leaf_strides[1],
                    page,
                    col < PAGES[g],
                )
            if tile == 0:
                length = tl.load(lengths + row).to(tl.int64)
                if COPIES[g]:
                    # A floor <= 1 means an unmodified copy in the router contract.
                    value = tl.maximum(length, FLOORS[g]) if FLOORS[g] > 1 else length
                    tl.store(
                        group_buffers.leaf_lengths
                        + row * group_buffers.leaf_strides[2],
                        value,
                    )
                for start in range(0, tokens_per_req, BLOCK):
                    step = start + tl.arange(0, BLOCK)
                    pos = tl.maximum(length - tokens_per_req + step, 0)
                    index = pos // PAGE_SIZES[g]
                    block = index // RATIOS[g]
                    window_raw = tl.load(
                        source + block * group_buffers.source_stride[1],
                        (row < actual_bs)
                        & (step < tokens_per_req)
                        & (index < PAGES[g])
                        & (block < group_buffers.columns),
                        other=0,
                    ).to(tl.int64)
                    window_page = window_raw * ATOM_PAGES[g] + index % RATIOS[g]
                    loc = tl.where(
                        window_raw > 0,
                        window_page * PAGE_SIZES[g] + pos % PAGE_SIZES[g],
                        0,
                    )
                    tl.store(
                        locations
                        + g * location_stride.to(tl.int64)
                        + row * tokens_per_req
                        + step,
                        loc,
                        step < tokens_per_req,
                    )


def prepare_decode_tables(
    sources: tuple[torch.Tensor, ...],
    tables: torch.Tensor,
    locations: torch.Tensor,
    lengths: torch.Tensor,
    *,
    actual_bs: int,
    tokens_per_req: int,
    ratios: tuple[int, ...],
    atom_pages_per_block: tuple[int, ...],
    pages: tuple[int, ...],
    page_sizes: tuple[int, ...],
    targets: tuple[tuple[torch.Tensor, torch.Tensor, int] | None, ...],
) -> None:
    """Fill existing decode buffers; return None.

    ``sources`` contains 2-D int32 scheduler tables in group order. ``tables``
    is the group's stacked int32 destination; ``locations`` stores the trailing
    ``tokens_per_req`` write slots per request. ``lengths`` defines the padded
    batch. Only ``actual_bs`` source rows are live. ``ratios`` counts logical
    subpages; ``atom_pages_per_block`` gives their physical spacing, including
    FlatKV padding. Both, ``pages`` and ``page_sizes`` are model-static geometry.
    Each target is None or a disjoint
    (length buffer, page table, length floor) tuple owned by its leaf. Destinations
    must not alias live sources or each other. Length floors <= 1 preserve the input.
    Non-CUDA tensors use the Torch reference; GPU implementations use the registry.
    """
    if not tables.is_cuda:
        from tokenspeed_kernel.numerics.torch.metadata import (
            torch_prepare_decode_tables,
        )

        torch_prepare_decode_tables(
            sources,
            tables,
            locations,
            lengths,
            actual_bs,
            tokens_per_req,
            ratios,
            atom_pages_per_block,
            pages,
            page_sizes,
            targets,
        )
        return
    kernel = select_kernel(
        "metadata",
        "prepare_decode_tables",
        format_signature(tables=dense_tensor_format(tables.dtype)),
    )
    kernel(
        sources,
        tables,
        locations,
        lengths,
        actual_bs,
        tokens_per_req,
        ratios,
        atom_pages_per_block,
        pages,
        page_sizes,
        targets,
    )


@register_kernel(
    "metadata",
    "prepare_decode_tables",
    name="triton_prepare_decode_tables",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("tables", "dense", {torch.int32}),
    priority=Priority.PORTABLE,
)
def triton_prepare_decode_tables(
    sources,
    tables,
    locations,
    lengths,
    actual_bs,
    tokens_per_req,
    ratios,
    atom_pages_per_block,
    pages,
    page_sizes,
    targets,
):
    """Portable implementation of :func:`prepare_decode_tables`."""
    groups = tuple(
        _GroupBuffers(
            source=source,
            source_stride=(source.stride(0), source.stride(1)),
            columns=source.shape[1],
            leaf_table=target[1] if target is not None else tables,
            leaf_lengths=target[0] if target is not None else lengths,
            leaf_strides=(
                (target[1].stride(0), target[1].stride(1), target[0].stride(0))
                if target is not None
                else (0, 0, 0)
            ),
        )
        for source, target in zip(sources, targets, strict=True)
    )
    _decode_tables_kernel[
        (lengths.numel(), len(ratios), triton.cdiv(tables.shape[2], 256))
    ](
        groups,
        tables,
        tables.stride(1),
        tables.stride(0),
        locations,
        locations.stride(0),
        lengths,
        actual_bs,
        tokens_per_req,
        RATIOS=ratios,
        ATOM_PAGES=atom_pages_per_block,
        PAGES=pages,
        PAGE_SIZES=page_sizes,
        FLOORS=tuple(t[2] if t is not None else 1 for t in targets),
        COPIES=tuple(t is not None for t in targets),
        CAPACITY=tables.shape[2],
        BLOCK=256,
    )
