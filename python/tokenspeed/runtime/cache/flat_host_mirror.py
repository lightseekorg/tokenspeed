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
tier (M15 Phase D). Transport mechanism only; scheduler/engine wiring is D2.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _identity_dedup(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    seen: dict[int, torch.Tensor] = {}
    for t in tensors:
        seen.setdefault(id(t), t)
    return list(seen.values())


def _state_slabs(device_kv_pool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(conv, ssm) state slab pairs, [] on pools predating state slabs."""
    return list(getattr(device_kv_pool, "state_slabs", None) or ())


def flat_bytes_per_host_page(device_kv_pool) -> int:
    """Bytes one host page occupies across all mirrors, computed from the
    device pool alone (no mirror allocation) -- the sizing side of
    ``FlatHostMirror.bytes_per_host_page`` for host-budget arithmetic.
    """
    tensors = _identity_dedup(device_kv_pool.k_buffer) + _identity_dedup(
        device_kv_pool.v_buffer
    )
    page_size = int(device_kv_pool.page_size)
    kv_bytes = sum(t.element_size() * t[0].numel() * page_size for t in tensors)
    # State slabs are page-indexed: one constant row per page id.
    state_bytes = sum(
        t.element_size() * t[0].numel()
        for pair in _state_slabs(device_kv_pool)
        for t in pair
    )
    return kv_bytes + state_bytes


class FlatHostMirror:
    """One pinned CPU mirror per DISTINCT device KV tensor plus one per
    state slab tensor; a (device_page, host_page) pair copies that page's
    row range on every mirror pair.

    Slab tensors are enumerated once each -- a page's rows are exactly its
    owner group's layers, so byte copies are group-safe by id-exclusivity.

    ``tensor_pairs`` order (PINNED, D2 fencing indexes into it): K*, V*,
    then state tensors flattened in slab order (conv0, ssm0, conv1, ...).
    KV mirrors span ``page_size`` token rows per page; state slabs are
    page-indexed (one snapshot row per page id), so their mirrors span 1
    row per page -- ``row_spans[i]`` carries each pair's span.
    """

    def __init__(self, device_kv_pool, num_host_pages: int):
        self.page_size = int(device_kv_pool.page_size)
        self.num_host_pages = int(num_host_pages)

        # Slab layout dedups 24 layer entries to 12 K + 12 V slabs; legacy
        # layout keeps all per-layer buffers (dead-row copies are harmless).
        k_tensors = _identity_dedup(device_kv_pool.k_buffer)
        v_tensors = _identity_dedup(device_kv_pool.v_buffer)
        self.num_k_tensors = len(k_tensors)

        k_index = {id(t): i for i, t in enumerate(k_tensors)}
        v_index = {id(t): i for i, t in enumerate(v_tensors)}
        self._layer_to_k_index = [
            k_index[id(t)] for t in device_kv_pool.k_buffer
        ]
        # Invariant D2 relies on: a layer's V tensor sits at
        # tensor_index_of_layer(layer) + num_k_tensors.
        assert self._layer_to_k_index == [
            v_index[id(t)] for t in device_kv_pool.v_buffer
        ], "flat host mirror: K/V dedup orders diverge"

        state_slabs = _state_slabs(device_kv_pool)
        state_tensors = [t for pair in state_slabs for t in pair]

        # layer -> slab pair index for state layers (identity-matched via
        # the pool's occurrence-indexed get_state_buffers binding).
        self._layer_to_state_pair: dict[int, int] = {}
        if state_slabs:
            pair_of_conv = {id(conv): n for n, (conv, _) in enumerate(state_slabs)}
            for layer_id in range(len(device_kv_pool.k_buffer)):
                try:
                    conv, _ssm = device_kv_pool.get_state_buffers(layer_id)
                except ValueError:
                    continue  # not a state layer
                self._layer_to_state_pair[layer_id] = pair_of_conv[id(conv)]

        pin = torch.cuda.is_available()
        kv_pairs = [
            (
                dev,
                torch.zeros(
                    (self.num_host_pages * self.page_size, *dev.shape[1:]),
                    dtype=dev.dtype,
                    pin_memory=pin,
                ),
            )
            for dev in k_tensors + v_tensors
        ]
        state_pairs = [
            (
                dev,
                torch.zeros(
                    (self.num_host_pages, *dev.shape[1:]),
                    dtype=dev.dtype,
                    pin_memory=pin,
                ),
            )
            for dev in state_tensors
        ]
        self.tensor_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (
            tuple(kv_pairs + state_pairs)
        )
        # Rows one page spans on each pair: page_size token rows for KV,
        # one page-indexed snapshot row for state slabs.
        self.row_spans: tuple[int, ...] = (self.page_size,) * len(kv_pairs) + (
            1,
        ) * len(state_pairs)

    def tensor_index_of_layer(self, layer_id: int) -> int:
        """Index of layer_id's K tensor in tensor_pairs (paired slab layers
        share the index); its V tensor is at index + num_k_tensors."""
        return self._layer_to_k_index[layer_id]

    def state_tensor_indices_of_layer(
        self, layer_id: int
    ) -> tuple[int, int] | None:
        """(conv_idx, ssm_idx) of layer_id's state slab pair in tensor_pairs
        (conv immediately precedes its ssm), or None for layers without
        state."""
        pair = self._layer_to_state_pair.get(layer_id)
        if pair is None:
            return None
        base = 2 * self.num_k_tensors + 2 * pair
        return base, base + 1

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
        order) after that tensor's copies -- D2's per-slab fencing hook."""
        return self._copy_pages(pairs, stream, to_host=False, record_events=True)
