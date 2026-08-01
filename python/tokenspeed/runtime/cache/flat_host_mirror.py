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

"""Byte-blind pinned-CPU mirror of a device KV pool for the flat L2 host
tier. Transport mechanism only; the executor owns scheduler/engine wiring.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HostMirrorFamily:
    """One mirrorable family of device tensors, declared per layer.

    The mirror stays byte-blind: a pool only says WHICH tensors carry a
    layer's page bytes and how many rows a page spans on them.

    Attributes:
        layer_tensors: Per layer, the family's tensors in mirror order;
            ``()`` for layers it does not cover (GDN state layers carry no
            KV). Tensors aliased across layers are mirrored once.
        rows_per_page: ``page_size`` for token-indexed buffers, 1 for
            page-indexed snapshots (state slabs).
    """

    layer_tensors: tuple[tuple[torch.Tensor, ...], ...]
    rows_per_page: int

    @classmethod
    def per_layer(
        cls, buffers: Sequence[torch.Tensor | None], rows_per_page: int
    ) -> HostMirrorFamily:
        """Family from one buffer (``None`` for an uncovered layer) per layer."""
        return cls(tuple(() if b is None else (b,) for b in buffers), rows_per_page)


def _flatten_families(
    families: Sequence[HostMirrorFamily],
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    """Flatten declared families into the mirrored tensor list.

    Walks families in declaration order, then layers, keeping the first
    occurrence of each distinct tensor (slab layouts alias one across paired
    layers).

    Args:
        families: The pool's ``host_mirror_families()``, all of one layer count.

    Returns:
        ``(tensors, row_spans, layer_fence)`` -- per layer, the fence is the
        index of the LAST mirrored tensor holding its bytes. Copies run in
        this order on one serial stream, so that event covers the earlier ones.

    Raises:
        ValueError: No families, differing layer counts, or a layer no family
            covers (nothing to fence it on).
    """
    if not families:
        raise ValueError("flat host mirror: the pool declares no tensor families")
    layer_num = len(families[0].layer_tensors)
    tensors: list[torch.Tensor] = []
    row_spans: list[int] = []
    index_of: dict[int, int] = {}
    fence = [-1] * layer_num  # -1 until some family claims the layer
    for family in families:
        if len(family.layer_tensors) != layer_num:
            raise ValueError(
                "flat host mirror: families disagree on layer count "
                f"({len(family.layer_tensors)} vs {layer_num})"
            )
        for layer_id, layer_tensors in enumerate(family.layer_tensors):
            for tensor in layer_tensors:
                index = index_of.get(id(tensor))
                if index is None:
                    index = len(tensors)
                    index_of[id(tensor)] = index
                    tensors.append(tensor)
                    row_spans.append(family.rows_per_page)
                fence[layer_id] = max(fence[layer_id], index)
    if -1 in fence:
        raise ValueError(
            "flat host mirror: layers "
            f"{[i for i, index in enumerate(fence) if index < 0]} are in no "
            "declared family, so a loadback could not fence them"
        )
    return tensors, row_spans, fence


def combined_host_mirror_families(
    device_kv_pool, draft_kv_pool=None
) -> list[HostMirrorFamily]:
    """Families to mirror for a (target, draft) KV pool pair.

    The drafter writes at the target's ``out_cache_loc`` and attends over the
    full sequence, so both pools share slot ids and one device page carries
    both. Restoring only the target would hand the drafter another request's
    KV -- output stays correct (the target verifies every draft token) but the
    acceptance rate collapses. Radix's ``KVPoolTransfer`` moves both for this.

    Draft families lead so every draft copy precedes every target copy on the
    serial load stream: the target's per-layer fences then cover them, and the
    drafter runs after the target forward has waited on those fences, so the
    draft pool needs no fence of its own.

    Args:
        device_kv_pool: The target KV pool.
        draft_kv_pool: The speculative draft KV pool, or None.

    Returns:
        Draft families (padded to the target's layer count) then the target's,
        or ``[]`` when the target declares none.

    Raises:
        ValueError: The draft pool declares no families, disagrees on page
            geometry, or has more layers than the target.
    """
    families = list(device_kv_pool.host_mirror_families())
    if draft_kv_pool is None or not families:
        return families
    layer_num = len(families[0].layer_tensors)
    draft_families = list(draft_kv_pool.host_mirror_families())
    if not draft_families:
        raise ValueError(
            f"flat host mirror: draft pool {type(draft_kv_pool).__name__} "
            "declares no tensor families, so its KV cannot be mirrored "
            "alongside the target's"
        )
    if int(draft_kv_pool.page_size) != int(device_kv_pool.page_size) or int(
        draft_kv_pool.size
    ) < int(device_kv_pool.size):
        raise ValueError(
            "flat host mirror: draft pool does not share the target's page "
            f"geometry (page_size {draft_kv_pool.page_size} vs "
            f"{device_kv_pool.page_size}, size {draft_kv_pool.size} vs "
            f"{device_kv_pool.size}); page ids would not line up"
        )
    padded: list[HostMirrorFamily] = []
    for family in draft_families:
        if len(family.layer_tensors) > layer_num:
            raise ValueError(
                "flat host mirror: draft pool has more layers than the target "
                f"({len(family.layer_tensors)} vs {layer_num})"
            )
        padded.append(
            HostMirrorFamily(
                family.layer_tensors + ((),) * (layer_num - len(family.layer_tensors)),
                family.rows_per_page,
            )
        )
    return padded + families


def flat_bytes_per_host_page(device_kv_pool, draft_kv_pool=None) -> int:
    """Bytes one host page occupies across all mirrors, computed from the
    device pools alone (no mirror allocation) -- the sizing side of
    ``FlatHostMirror.bytes_per_host_page`` for host-budget arithmetic.
    """
    tensors, row_spans, _ = _flatten_families(
        combined_host_mirror_families(device_kv_pool, draft_kv_pool)
    )
    return sum(
        t.element_size() * t[0].numel() * span for t, span in zip(tensors, row_spans)
    )


class FlatHostMirror:
    """One pinned CPU mirror per DISTINCT device tensor the pool declares in
    ``host_mirror_families()``; a (device_page, host_page) pair copies that
    page's row range on every mirror pair.

    Aliased slab tensors are enumerated once each -- a page's rows are
    exactly its owner group's layers, so byte copies are group-safe by
    id-exclusivity.

    ``tensor_pairs`` follows the declared family order (the executor's fencing indexes
    into it): for the base K/V layout that is K*, V*, then state tensors in
    slab order (conv0, ssm0, conv1, ...); MLA declares its fused latent
    instead, DSA appends packed index-K. A speculative draft pool's families
    lead the whole list (see ``combined_host_mirror_families``).
    Token-indexed mirrors span ``page_size`` rows per page, page-indexed ones
    (state slabs) span 1 -- ``row_spans[i]`` carries each pair's span.
    """

    def __init__(self, device_kv_pool, num_host_pages: int, draft_kv_pool=None):
        self.page_size = int(device_kv_pool.page_size)
        self.num_host_pages = int(num_host_pages)

        tensors, row_spans, self._layer_fence = _flatten_families(
            combined_host_mirror_families(device_kv_pool, draft_kv_pool)
        )
        self.layer_num = len(self._layer_fence)

        pin = torch.cuda.is_available()
        self.tensor_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = tuple(
            (
                dev,
                torch.zeros(
                    (self.num_host_pages * span, *dev.shape[1:]),
                    dtype=dev.dtype,
                    pin_memory=pin,
                ),
            )
            for dev, span in zip(tensors, row_spans)
        )
        self.row_spans: tuple[int, ...] = tuple(row_spans)

    def fence_tensor_index_of_layer(self, layer_id: int) -> int:
        """Index in ``tensor_pairs`` of the LAST mirror carrying layer_id's
        bytes: once that tensor's copy lands the layer is fully readable
        (the load stream is serial, so the event covers its earlier copies).
        Paired slab layers share an index -- correct by design."""
        return self._layer_fence[layer_id]

    def bytes_per_host_page(self) -> int:
        return sum(
            dev.element_size() * dev[0].numel() * span
            for (dev, _), span in zip(self.tensor_pairs, self.row_spans)
        )

    def _copy_pages(
        self,
        pairs: Iterable[tuple[int, int]],
        stream,
        to_host: bool,
        record_events: bool,
    ) -> list[torch.cuda.Event]:
        pairs = list(pairs)
        events: list[torch.cuda.Event] = []
        with torch.cuda.stream(stream):
            for (dev, mirror), p in zip(self.tensor_pairs, self.row_spans):
                for device_page, host_page in pairs:
                    dev_rows = dev[device_page * p : (device_page + 1) * p]
                    host_rows = mirror[host_page * p : (host_page + 1) * p]
                    if to_host:
                        host_rows.copy_(dev_rows, non_blocking=True)
                    else:
                        dev_rows.copy_(host_rows, non_blocking=True)
                if record_events:
                    event = torch.cuda.Event()
                    event.record()
                    events.append(event)
        return events

    def store_pages(self, pairs: Iterable[tuple[int, int]], stream) -> None:
        """Copy each (device_page, host_page) pair device -> host on stream."""
        self._copy_pages(pairs, stream, to_host=True, record_events=False)

    def load_pages(self, pairs: Iterable[tuple[int, int]], stream) -> None:
        """Copy each (device_page, host_page) pair host -> device on stream."""
        self._copy_pages(pairs, stream, to_host=False, record_events=False)

    def load_pages_with_events(
        self, pairs: Iterable[tuple[int, int]], stream
    ) -> list[torch.cuda.Event]:
        """load_pages, recording one event per device tensor (tensor_pairs
        order) after that tensor's copies -- the executor's per-slab fencing hook."""
        return self._copy_pages(pairs, stream, to_host=False, record_events=True)
