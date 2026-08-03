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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import torch


def _page_ratio(logical_page_size: int, kernel_page_size: int) -> int:
    if (
        logical_page_size <= 0
        or kernel_page_size <= 0
        or logical_page_size % kernel_page_size
    ):
        raise ValueError(
            "logical_page_size must be a positive multiple of kernel_page_size, "
            f"got {logical_page_size} and {kernel_page_size}"
        )
    return logical_page_size // kernel_page_size


def expand_page_table(
    page_table: torch.Tensor,
    *,
    logical_page_size: int,
    kernel_page_size: int,
    max_kernel_pages: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand scheduler page IDs into the smaller pages consumed by a kernel."""
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be 2-D, got shape {tuple(page_table.shape)}")
    ratio = _page_ratio(logical_page_size, kernel_page_size)
    if max_kernel_pages is None:
        max_kernel_pages = page_table.shape[1] * ratio
    if max_kernel_pages < 0:
        raise ValueError(
            f"max_kernel_pages must be non-negative, got {max_kernel_pages}"
        )

    rows = page_table.shape[0]
    if ratio == 1 and out is None and max_kernel_pages <= page_table.shape[1]:
        return page_table[:, :max_kernel_pages]

    if out is None:
        out = torch.zeros(
            (rows, max_kernel_pages),
            dtype=page_table.dtype,
            device=page_table.device,
        )
    else:
        if out.ndim != 2 or out.shape[0] < rows or out.shape[1] < max_kernel_pages:
            raise ValueError(
                f"out shape {tuple(out.shape)} cannot hold ({rows}, {max_kernel_pages})"
            )
        if out.dtype != page_table.dtype or out.device != page_table.device:
            raise ValueError("out must have the same dtype and device as page_table")
        out[:rows].zero_()

    result = out[:rows, :max_kernel_pages]
    if ratio == 1:
        copy_columns = min(max_kernel_pages, page_table.shape[1])
        result[:, :copy_columns].copy_(page_table[:, :copy_columns])
        return result

    logical_columns = min(
        page_table.shape[1],
        (max_kernel_pages + ratio - 1) // ratio,
    )
    if logical_columns == 0:
        return result

    kernel_offsets = torch.arange(
        ratio,
        dtype=page_table.dtype,
        device=page_table.device,
    )
    expanded = (
        page_table[:, :logical_columns].clamp_min(0).unsqueeze(-1) * ratio
        + kernel_offsets
    ).reshape(rows, logical_columns * ratio)
    copy_columns = min(max_kernel_pages, expanded.shape[1])
    result[:, :copy_columns].copy_(expanded[:, :copy_columns])
    return result
