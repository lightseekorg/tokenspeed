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

"""Pack strided Prefill KV rows into contiguous SGEs before Mooncake WRITE.

Heterogeneous TP partitions heads, so a page is ``rows_per_page`` rows of
``bytes_per_row`` with a larger source pitch. Decode's local layout is often
already contiguous (``dst_stride == bytes_per_row``). Expanding each row into
its own 1KB SGE dominates RDMA submit time; a device-side 2D copy plus one
SGE matches equal-TP descriptor counts.

Destination-strided fragments still expand to per-row SGEs: a contiguous
WRITE would land on the wrong Decode layout without a remote unpack.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

Sge = tuple[int, int, int]


@dataclass(frozen=True)
class PackedCopy:
    """One dest-contiguous, src-strided page fragment to pack before WRITE."""

    src: int
    dst: int
    width: int
    src_pitch: int
    rows: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.src_pitch < self.width or self.rows <= 0:
            raise ValueError(
                f"invalid PackedCopy width={self.width} src_pitch={self.src_pitch} "
                f"rows={self.rows}"
            )

    @property
    def nbytes(self) -> int:
        return self.width * self.rows


@dataclass(frozen=True)
class PageFieldCopies:
    """One cache group's whole-field copies as pages x fields, unexpanded.

    ``fields`` is a C-contiguous int64 ``[F, 5]`` table of
    ``(src_base, src_stride, dst_base, dst_stride, length)`` rows; descriptor
    ``(f, i)`` copies ``length`` bytes from ``src_base + src_pages[i] *
    src_stride`` to ``dst_base + dst_pages[i] * dst_stride``. Mooncake's
    ``batch_transfer_sync_write_pages`` consumes this shape directly, so a
    group is one call with no descriptor built on the host. Never packed.
    """

    src_pages: np.ndarray
    dst_pages: np.ndarray
    fields: np.ndarray

    def __post_init__(self) -> None:
        for name, column in (
            ("src_pages", self.src_pages),
            ("dst_pages", self.dst_pages),
        ):
            if (
                not isinstance(column, np.ndarray)
                or column.ndim != 1
                or column.dtype != np.int64
            ):
                raise TypeError(f"PageFieldCopies.{name} must be a 1-D int64 array")
        if self.src_pages.shape != self.dst_pages.shape:
            raise ValueError("PageFieldCopies page arrays must have the same length")
        if (
            not isinstance(self.fields, np.ndarray)
            or self.fields.ndim != 2
            or self.fields.shape[1] != 5
            or self.fields.dtype != np.int64
            or not self.fields.flags.c_contiguous
        ):
            raise TypeError(
                "PageFieldCopies.fields must be a C-contiguous int64 [F, 5] array"
            )

    def __len__(self) -> int:
        return int(self.src_pages.shape[0] * self.fields.shape[0])


def expand_packed_copy(copy: PackedCopy) -> list[Sge]:
    """Row-wise SGEs used when 2D pack is unavailable. Dest is contiguous."""
    return [
        (
            copy.src + row * copy.src_pitch,
            copy.dst + row * copy.width,
            copy.width,
        )
        for row in range(copy.rows)
    ]


def flatten_transfer_blocks(blocks: Iterable[object]) -> Iterator[Sge]:
    """Yield row SGEs for PackedCopy entries; pass 3-tuples through.

    Expansion is lazy so the WRITE path can flush every 4096 descriptors
    even when a packed page becomes many rows.
    """
    for block in blocks:
        if isinstance(block, PackedCopy):
            yield from expand_packed_copy(block)
            continue
        if isinstance(block, PageFieldCopies):
            raise TypeError("PageFieldCopies takes the page-gathered WRITE, not SGEs")
        src, dst, length = block
        yield (int(src), int(dst), int(length))


def coalesce_transfer_blocks(blocks: Iterable[Sge]) -> Iterator[Sge]:
    """Merge adjacent byte ranges contiguous at both source and destination.

    Input and output entries are (source address, destination address, length).
    Preserve order and gaps; consume only one lookahead entry. Call within a
    single peer's ready transfer batch, after packed copies are materialized.
    """
    pending: Sge | None = None
    for src, dst, length in blocks:
        if pending is not None:
            prev_src, prev_dst, prev_length = pending
            if src == prev_src + prev_length and dst == prev_dst + prev_length:
                pending = (prev_src, prev_dst, prev_length + length)
                continue
            yield pending
        pending = (src, dst, length)
    if pending is not None:
        yield pending
