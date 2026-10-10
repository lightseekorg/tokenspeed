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

An owner that keys its slots by request -- the sampling backends' pools, the
DSpark windows -- prepares a slot at the request's first forward and skips
that reset while the slot's recorded request id matches. Such an owner images
a **prepared marker** in front of its rows: a victim that never ran a forward
on this engine (a PD decode role's request retracted between its landing and
its first decode) still holds the slot's previous occupant's rows, so the
image carries the marker as not-prepared and no rows, and the restore leaves
the new slot unclaimed for the first forward to prepare from the request's own
parameters -- exactly what the unretracted request would have met.

Every owner lists its per-slot tensors once, as ``slot_state_rows(slot)``,
and the two copies and the size derive from that list, so an owner cannot
export a row it does not import. The architecture test in
``test/runtime/test_slot_state.py`` enumerates every per-slot tensor of the
implementing classes and requires each to be either exported here or named
token-derived.

The blob's layout -- which owner's segment sits where -- is fixed for the
executor's lifetime, so :class:`SlotStateLayout` computes it once when the
Host cache executor is built; a victim's export and a restore's import only
slice at the recorded offsets.
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

#: The prepared marker of a request-keyed owner: one alignment unit in front
#: of its rows, byte 0 being 1 when the imaged slot was prepared for the
#: imaged request (``write_prepared_marker`` / ``read_prepared_marker``).
PREPARED_MARKER_BYTES = SLOT_STATE_ALIGNMENT


@runtime_checkable
class SlotStateExporter(Protocol):
    """One owner of per-slot Device state outside the cache groups."""

    def slot_state_bytes(self) -> int:
        """Bytes of one slot's image; fixed for the executor's lifetime."""

    def export_slot_state(
        self, slot: int, out: torch.Tensor, stream, *, request_id: str
    ) -> None:
        """Copy ``slot``'s state into ``out`` (uint8, ``slot_state_bytes()`` long).

        Enqueued on ``stream``; the caller orders that stream behind the
        stream the state was written on and records the completion event.
        ``request_id`` is the victim the slot is imaged for: an owner that
        keys its slots by request images whether the slot was prepared for
        it (the module docstring).
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


def stream_scope(stream):
    """Enqueue on ``stream`` (a transfer stream), or inline for a CPU-only owner."""
    return nullcontext() if stream is None else device_module.stream(stream)


def _image_views(rows: Sequence[torch.Tensor], image: torch.Tensor):
    """Pair each row with its typed view into ``image``; raise on overflow.

    One pass over the rows: the offsets fall out of forming the views.
    """
    if image.dtype != torch.uint8 or image.ndim != 1:
        raise ValueError("a slot-state image is a 1-D uint8 tensor")
    capacity = image.numel()
    offset = 0
    views = []
    for row in rows:
        nbytes = row.numel() * row.element_size()
        if offset + nbytes > capacity:
            break
        view = image[offset : offset + nbytes].view(row.dtype).view(row.shape)
        views.append((row, view))
        offset += aligned_slot_state_bytes(nbytes)
    # The image must hold the padded rows too: a segment is always a whole
    # number of alignment units.
    if len(views) < len(rows) or offset > capacity:
        raise ValueError(
            f"slot-state image of {capacity} bytes cannot hold "
            f"{slot_state_image_bytes(rows)} bytes"
        )
    return views


def write_prepared_marker(image: torch.Tensor, prepared: bool) -> None:
    """Write the prepared marker at the head of ``image`` (a CPU write)."""
    if image.numel() < PREPARED_MARKER_BYTES:
        raise ValueError("a slot-state image is too short for the prepared marker")
    image[0] = 1 if prepared else 0


def read_prepared_marker(image: torch.Tensor) -> bool:
    """Whether the image's head marker says the slot was prepared."""
    if image.numel() < PREPARED_MARKER_BYTES:
        raise ValueError("a slot-state image is too short for the prepared marker")
    return bool(image[0].item())


def pack_slot_rows(rows: Sequence[torch.Tensor], image: torch.Tensor, stream) -> None:
    """Copy one slot's ``rows`` into ``image`` on ``stream`` (Device to Host).

    Args:
        rows: The slot's views, in the owner's fixed order.
        image: The pinned uint8 image row; at least
            :func:`slot_state_image_bytes` of ``rows`` long.
        stream: The transfer stream, or None for a CPU-only owner.
    """
    views = _image_views(rows, image)
    with stream_scope(stream):
        for row, view in views:
            view.copy_(row, non_blocking=True)


def unpack_slot_rows(rows: Sequence[torch.Tensor], image: torch.Tensor, stream) -> None:
    """Copy ``image`` back into one slot's ``rows`` on ``stream`` (Host to Device).

    Arguments as for :func:`pack_slot_rows`.
    """
    views = _image_views(rows, image)
    with stream_scope(stream):
        for row, view in views:
            row.copy_(view, non_blocking=True)


class SlotStateLayout:
    """The fixed byte layout of one slot's image: one segment per exporter.

    Built once, when the Host cache executor is constructed: every exporter's
    size is fixed for the executor's lifetime, so the per-victim export and
    the per-restore import slice ``out`` / ``src`` at the recorded offsets and
    hand each exporter its segment -- no size queries on the forward thread.
    """

    __slots__ = ("nbytes", "segments")

    def __init__(self, exporters: Sequence[SlotStateExporter]) -> None:
        """
        Args:
            exporters: The owners in blob order; each contributes
                ``slot_state_bytes()`` bytes, which must be alignment-padded.
        """
        segments: list[tuple[SlotStateExporter, int, int]] = []
        offset = 0
        for exporter in exporters:
            size = int(exporter.slot_state_bytes())
            if size % SLOT_STATE_ALIGNMENT:
                raise ValueError(
                    f"{type(exporter).__name__} slot-state segment of {size} bytes "
                    f"is not padded to {SLOT_STATE_ALIGNMENT}"
                )
            segments.append((exporter, offset, offset + size))
            offset += size
        #: ``(exporter, begin, end)`` byte ranges, in blob order.
        self.segments = tuple(segments)
        #: The blob width: the snapshot arena's row size.
        self.nbytes = offset

    def _check(self, image: torch.Tensor) -> None:
        if image.numel() < self.nbytes:
            raise ValueError(
                f"slot-state exporters need {self.nbytes} bytes, image has "
                f"{image.numel()}"
            )

    def export(self, slot: int, out: torch.Tensor, stream, *, request_id: str) -> None:
        """Image ``slot``, ``request_id``'s, into ``out`` on ``stream``, one
        segment per exporter."""
        self._check(out)
        for exporter, begin, end in self.segments:
            exporter.export_slot_state(
                slot, out[begin:end], stream, request_id=request_id
            )

    def import_(self, slot: int, src: torch.Tensor, stream, *, request_id: str) -> None:
        """Restore ``src`` into ``slot``, now ``request_id``'s, on ``stream``."""
        self._check(src)
        for exporter, begin, end in self.segments:
            exporter.import_slot_state(
                slot, src[begin:end], stream, request_id=request_id
            )
