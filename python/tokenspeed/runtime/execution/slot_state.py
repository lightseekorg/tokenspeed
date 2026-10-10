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

"""The slot-state image: per-request Device state outside every cache group.

A retraction snapshot copies a request's cache-group pages to a pinned Host
image and restores them into fresh pages later. Device state keyed by the
request's pool slot (``req_pool_index``) but stored in no cache group -- the
next step's inputs, the drafter's stash, a recurrent ring, a partial index
pool -- must travel with the pages or the continuation is not exact. Each
owner of such state implements :class:`SlotStateExporter`; the model executor
concatenates the owners into one blob per slot and the snapshot executor keeps
a pinned arena of those blobs.

Token-derived rows are deliberately NOT part of the image: the request's
committed-token history and the n-gram tail are reseeded from the control
plane's token list on a slot change, exactly as a slot handoff is today.

Every owner lists its per-slot tensors once, as ``slot_state_rows(slot)``,
and the two copies and the size derive from that list, so an owner cannot
export a row it does not import. The architecture test in
``test/runtime/test_slot_state.py`` enumerates every per-slot tensor of the
implementing classes and requires each to be either exported here or named
token-derived.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from typing import Protocol, runtime_checkable

import torch

from tokenspeed.runtime.utils import get_device_module

device_module = get_device_module()

#: Every row of the image starts on this byte boundary so a typed view of
#: the uint8 blob is always aligned, whatever the dtypes before it.
SLOT_STATE_ALIGNMENT = 16


@runtime_checkable
class SlotStateExporter(Protocol):
    """One owner of per-slot Device state outside the cache groups."""

    def slot_state_bytes(self) -> int:
        """Bytes of one slot's image; fixed for the executor's lifetime."""

    def export_slot_state(self, slot: int, out: torch.Tensor, stream) -> None:
        """Copy ``slot``'s state into ``out`` (uint8, ``slot_state_bytes()`` long).

        Enqueued on ``stream``; the caller orders that stream behind the
        stream the state was written on and records the completion event.
        """

    def import_slot_state(
        self, slot: int, src: torch.Tensor, stream, *, request_id: str
    ) -> None:
        """Copy an image back into ``slot`` for ``request_id``.

        ``request_id`` lets an owner that keys its slots by request (DSpark's
        windows) claim the slot, so its next prologue does not treat the
        restored rows as a stale occupant's.
        """


def aligned_slot_state_bytes(nbytes: int) -> int:
    """``nbytes`` rounded up to the image alignment."""
    return -(-nbytes // SLOT_STATE_ALIGNMENT) * SLOT_STATE_ALIGNMENT


def slot_state_image_bytes(rows: Iterable[torch.Tensor]) -> int:
    """The image size of one slot's ``rows``, each padded to the alignment."""
    return sum(
        aligned_slot_state_bytes(row.numel() * row.element_size()) for row in rows
    )


def _stream_scope(stream):
    return nullcontext() if stream is None else device_module.stream(stream)


def _image_views(rows: Sequence[torch.Tensor], image: torch.Tensor):
    """Pair each row with its typed view into ``image``; raise on overflow."""
    if image.dtype != torch.uint8 or image.ndim != 1:
        raise ValueError("a slot-state image is a 1-D uint8 tensor")
    needed = slot_state_image_bytes(rows)
    if needed > image.numel():
        raise ValueError(
            f"slot-state image of {image.numel()} bytes cannot hold {needed} bytes"
        )
    offset = 0
    views = []
    for row in rows:
        nbytes = row.numel() * row.element_size()
        view = image[offset : offset + nbytes].view(row.dtype).view(row.shape)
        views.append((row, view))
        offset += aligned_slot_state_bytes(nbytes)
    return views


def pack_slot_rows(rows: Sequence[torch.Tensor], image: torch.Tensor, stream) -> None:
    """Copy one slot's ``rows`` into ``image`` on ``stream`` (Device to Host).

    Args:
        rows: The slot's views, in the owner's fixed order.
        image: The pinned uint8 image row; at least
            :func:`slot_state_image_bytes` of ``rows`` long.
        stream: The transfer stream, or None for a CPU-only owner.
    """
    views = _image_views(rows, image)
    with _stream_scope(stream):
        for row, view in views:
            view.copy_(row, non_blocking=True)


def unpack_slot_rows(rows: Sequence[torch.Tensor], image: torch.Tensor, stream) -> None:
    """Copy ``image`` back into one slot's ``rows`` on ``stream`` (Host to Device).

    Arguments as for :func:`pack_slot_rows`.
    """
    views = _image_views(rows, image)
    with _stream_scope(stream):
        for row, view in views:
            row.copy_(view, non_blocking=True)


def _segments(
    exporters: Sequence[SlotStateExporter], image: torch.Tensor
) -> list[tuple[SlotStateExporter, torch.Tensor]]:
    """Slice ``image`` into one segment per exporter, in order; raise on overflow."""
    sizes = [exporter.slot_state_bytes() for exporter in exporters]
    if any(size % SLOT_STATE_ALIGNMENT for size in sizes):
        raise ValueError("every slot-state segment must be alignment-padded")
    if sum(sizes) > image.numel():
        raise ValueError(
            f"slot-state exporters need {sum(sizes)} bytes, image has {image.numel()}"
        )
    segments = []
    offset = 0
    for exporter, size in zip(exporters, sizes):
        segments.append((exporter, image[offset : offset + size]))
        offset += size
    return segments


def export_slot_state_sequence(
    exporters: Sequence[SlotStateExporter], slot: int, out: torch.Tensor, stream
) -> None:
    """Export ``slot`` through ``exporters`` into consecutive segments of ``out``."""
    for exporter, segment in _segments(exporters, out):
        exporter.export_slot_state(slot, segment, stream)


def import_slot_state_sequence(
    exporters: Sequence[SlotStateExporter],
    slot: int,
    src: torch.Tensor,
    stream,
    *,
    request_id: str,
) -> None:
    """Import consecutive segments of ``src`` into ``slot`` through ``exporters``."""
    for exporter, segment in _segments(exporters, src):
        exporter.import_slot_state(slot, segment, stream, request_id=request_id)
