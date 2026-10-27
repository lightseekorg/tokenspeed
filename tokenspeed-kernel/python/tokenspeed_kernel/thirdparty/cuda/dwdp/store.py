# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Immutable peer weights with composite views and two asynchronous copy slots.

The composite mapping follows TensorRT-LLM DWDP. Peer mappings are grouped by
owner so the number of fabric access regions does not grow as layers × peers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.peer_weights import (
    ExpertPartition,
    ExpertWeightShard,
)
from tokenspeed_kernel.thirdparty.cuda.dwdp import vmm
from tokenspeed_kernel.thirdparty.cuda.dwdp.layout import WeightLayout


@dataclass
class _Field:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    expert_bytes: int
    layout: WeightLayout | None
    handle: int | None
    address: int | None
    replica: torch.Tensor | None


@dataclass
class _Layer:
    index: int
    slot: int
    layout: str
    fields: tuple[_Field, ...]


@dataclass
class _Slot:
    ready: torch.cuda.Event
    released: torch.cuda.Event
    layer: int | None
    last_reader_stream: int | None


class CudaPeerWeightStore:
    def __init__(
        self,
        *,
        partition: ExpertPartition,
        group: dist.ProcessGroup,
        device: torch.device,
        model_identity: str,
        pool_page_bytes: int,
        copy_implementation: str,
    ) -> None:
        if copy_implementation != "simple":
            raise ValueError(
                f"Unsupported peer copy implementation: {copy_implementation}"
            )
        if dist.get_backend(group) != "gloo":
            raise ValueError("Weight setup requires a CPU gloo group")
        if device.type != "cuda" or device.index is None:
            raise ValueError("Peer weights require an explicit CUDA device index")
        if device.index != torch.cuda.current_device():
            raise ValueError("Peer weights must be constructed on the current device")
        if group.rank() != partition.rank or group.size() != partition.size:
            raise ValueError("Weight ownership and process-group ranks disagree")
        self.partition = partition
        self.group = group
        self.device = device
        self.model_identity = model_identity
        self.pool_page_bytes = pool_page_bytes
        self.granularity = vmm.granularity(device.index)
        if pool_page_bytes <= 0 or pool_page_bytes % self.granularity:
            raise ValueError("Staging page size must be a multiple of VMM granularity")
        self._stream = torch.cuda.Stream(device=device)
        self._published = torch.cuda.Event()
        self._published.record(torch.cuda.current_stream(device))
        self._slots = [
            _Slot(torch.cuda.Event(), torch.cuda.Event(), None, None) for _ in range(2)
        ]
        for slot in self._slots:
            slot.released.record(torch.cuda.current_stream(device))
        self._layers: dict[int, _Layer] = {}
        self._views: dict[int, dict[str, torch.Tensor]] = {}
        self._plans: dict[int, list[tuple[int, int, int]]] = {}
        self._mappings: list[tuple[int, int]] = []
        self._reservations: list[tuple[int, int]] = []
        self._handles: list[int] = []
        self._physical_bytes = 0
        self._staging_bytes = 0
        self._reader: int | None = None
        self._reader_stream: int | None = None
        self._finalized = False
        self._closed = False
        self._view_owner = object()

    @property
    def physical_bytes(self) -> int:
        return self._physical_bytes

    @property
    def staging_bytes(self) -> int:
        return self._staging_bytes

    def _allocate(self, size: int, *, fabric: bool) -> int:
        handle = vmm.allocate(size, self.device.index, fabric=fabric)
        self._handles.append(handle)
        self._physical_bytes += size
        return handle

    def _reserve(self, size: int) -> int:
        address = vmm.reserve(size, self.granularity)
        self._reservations.append((address, size))
        return address

    def _map(self, address: int, size: int, handle: int) -> None:
        vmm.map_handle(address, size, handle)
        self._mappings.append((address, size))

    def _layout(self, expert_bytes: int, rank: int) -> WeightLayout:
        return WeightLayout.create(
            expert_bytes=expert_bytes,
            experts=self.partition.num_experts,
            first=rank * self.partition.count,
            count=self.partition.count,
            granularity=self.granularity,
            pool_page_bytes=self.pool_page_bytes,
        )

    def publish_layer(
        self, layer: int, shards: tuple[ExpertWeightShard, ...], *, layout: str
    ) -> dict[str, torch.Tensor]:
        if self._closed or self._finalized or layer in self._layers:
            raise RuntimeError(
                f"Cannot publish layer {layer} in the current store state"
            )
        if not shards or len({s.name for s in shards}) != len(shards):
            raise ValueError("A layer must contain uniquely named expert tensors")
        for shard in shards:
            value = shard.tensor
            if (
                value.ndim == 0
                or value.shape[0] != self.partition.count
                or value.device != self.device
                or not value.is_contiguous()
            ):
                raise ValueError(f"Invalid owned expert tensor {shard.name}")
        torch.cuda.current_stream(self.device).wait_event(self._published)
        fields = []
        result = {}
        for shard in shards:
            value = shard.tensor.detach()
            expert_bytes = value[0].numel() * value.element_size()
            if shard.replicate:
                fields.append(
                    _Field(
                        shard.name,
                        tuple(value.shape),
                        value.dtype,
                        expert_bytes,
                        None,
                        None,
                        None,
                        value.cpu(),
                    )
                )
                result[shard.name] = value
                continue
            geometry = self._layout(expert_bytes, self.partition.rank)
            handle = self._allocate(geometry.source_bytes, fabric=True)
            address = self._reserve(geometry.source_bytes)
            self._map(address, geometry.source_bytes, handle)
            vmm.set_access(address, geometry.source_bytes, self.device.index)
            owned = vmm.tensor_view(
                address + geometry.source_offset,
                tuple(value.shape),
                value.dtype,
                self.device,
                self._view_owner,
            )
            owned.copy_(value)
            self._published.record(torch.cuda.current_stream(self.device))
            result[shard.name] = owned
            fields.append(
                _Field(
                    shard.name,
                    tuple(value.shape),
                    value.dtype,
                    expert_bytes,
                    geometry,
                    handle,
                    address,
                    None,
                )
            )
        self._layers[layer] = _Layer(
            layer, len(self._layers) % 2, layout, tuple(fields)
        )
        self._published.record(torch.cuda.current_stream(self.device))
        return result

    def _descriptors(self) -> tuple:
        return (
            self.model_identity,
            self.partition.num_experts,
            self.partition.size,
            self.pool_page_bytes,
            self.granularity,
            tuple(
                (
                    layer.index,
                    layer.layout,
                    tuple(
                        (f.name, f.shape, str(f.dtype), f.layout is None)
                        for f in layer.fields
                    ),
                )
                for layer in self._layers.values()
            ),
        )

    def _peer_sources(self, payloads: list[dict]) -> dict[tuple[int, int, str], int]:
        addresses = {}
        for peer, payload in enumerate(payloads):
            streamed = [
                (layer.index, f, self._layout(f.expert_bytes, peer))
                for layer in self._layers.values()
                for f in layer.fields
                if f.layout is not None
            ]
            if peer == self.partition.rank:
                for index, field, _ in streamed:
                    addresses[peer, index, field.name] = field.address
                continue
            size = sum(geometry.source_bytes for _, _, geometry in streamed)
            if not size:
                continue
            arena = self._reserve(size)
            offset = 0
            for index, field, geometry in streamed:
                handle = vmm.import_handle(payload["handles"][index, field.name])
                self._handles.append(handle)
                self._map(arena + offset, geometry.source_bytes, handle)
                addresses[peer, index, field.name] = arena + offset
                offset += geometry.source_bytes
            # One access region per peer avoids exhausting fabric routing entries.
            vmm.set_access(arena, size, self.device.index)
        return addresses

    def _copies(
        self,
        layer: _Layer,
        field: _Field,
        intervals: tuple[tuple[int, int], ...],
        destination: int,
        peer_addresses: dict[tuple[int, int, str], int],
    ) -> list[tuple[int, int, int]]:
        copies = []
        for step in range(1, self.partition.size):
            peer = (self.partition.rank + step) % self.partition.size
            geometry = self._layout(field.expert_bytes, peer)
            source = peer_addresses[peer, layer.index, field.name]
            for begin, end in intervals:
                begin = max(begin, geometry.local_begin)
                end = min(end, geometry.local_end)
                if begin < end:
                    copies.append(
                        (
                            destination + begin,
                            source + begin - geometry.page_begin,
                            end - begin,
                        )
                    )
        return copies

    def finalize(self) -> None:
        if self._closed or self._finalized or not self._layers:
            raise RuntimeError("A nonempty unpublished store is required for finalize")
        self._published.synchronize()
        payload = {
            "descriptor": self._descriptors(),
            "handles": {
                (layer.index, f.name): vmm.export(f.handle)
                for layer in self._layers.values()
                for f in layer.fields
                if f.handle is not None
            },
            "replicas": {
                (layer.index, f.name): f.replica
                for layer in self._layers.values()
                for f in layer.fields
                if f.replica is not None
            },
        }
        payloads = [None] * self.partition.size
        dist.all_gather_object(payloads, payload, group=self.group)
        if any(p["descriptor"] != payload["descriptor"] for p in payloads):
            raise ValueError(
                "Peer checkpoint, tensor layout, or ownership metadata differs"
            )
        peers = self._peer_sources(payloads)
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            for field in layer.fields:
                if field.layout is None:
                    continue
                # Only immutable boundary bytes share the owner's physical pages.
                for dst, src, size in self._copies(
                    layer,
                    field,
                    field.layout.edge_intervals(),
                    field.address - field.layout.page_begin,
                    peers,
                ):
                    vmm.copy(dst, src, size, stream)
        stream.synchronize()
        dist.barrier(group=self.group)
        slot_pages = [0, 0]
        for layer in self._layers.values():
            slot_pages[layer.slot] = max(
                slot_pages[layer.slot],
                sum(
                    f.layout.staging_pages for f in layer.fields if f.layout is not None
                ),
            )
        pools = [
            [self._allocate(self.pool_page_bytes, fabric=False) for _ in range(n)]
            for n in slot_pages
        ]
        self._staging_bytes = sum(slot_pages) * self.pool_page_bytes
        for layer in self._layers.values():
            views = {}
            plan = []
            page_index = 0
            for field in layer.fields:
                shape = (self.partition.num_experts, *field.shape[1:])
                if field.layout is None:
                    value = torch.cat(
                        [p["replicas"][layer.index, field.name] for p in payloads]
                    ).to(self.device)
                    views[field.name] = value
                    self._physical_bytes += value.numel() * value.element_size()
                    field.replica = None
                    continue
                geometry = field.layout
                address = self._reserve(geometry.virtual_bytes)
                for begin, length in (
                    (0, geometry.pre_bytes),
                    (geometry.pre_bytes + geometry.source_bytes, geometry.post_bytes),
                ):
                    for offset in range(begin, begin + length, self.pool_page_bytes):
                        self._map(
                            address + offset,
                            self.pool_page_bytes,
                            pools[layer.slot][page_index],
                        )
                        page_index += 1
                self._map(
                    address + geometry.pre_bytes, geometry.source_bytes, field.handle
                )
                vmm.set_access(address, geometry.virtual_bytes, self.device.index)
                pointer = address + geometry.view_offset
                views[field.name] = vmm.tensor_view(
                    pointer, shape, field.dtype, self.device, self._view_owner
                )
                plan.extend(
                    self._copies(
                        layer, field, geometry.remote_intervals(), pointer, peers
                    )
                )
            self._views[layer.index] = views
            self._plans[layer.index] = plan
        stream.synchronize()
        self._finalized = True

    def _require_ready(self) -> None:
        if not self._finalized or self._closed:
            raise RuntimeError("Peer weights are not available")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Peer-weight fetches must be outside CUDA graph capture")

    def weights(self, layer: int) -> dict[str, torch.Tensor]:
        if not self._finalized or self._closed:
            raise RuntimeError("Peer weights are not available")
        return self._views[layer]

    def prefetch(self, layer: int) -> None:
        self._require_ready()
        slot = self._slots[self._layers[layer].slot]
        if slot.layer == layer:
            return
        if (
            self._reader is not None
            and self._layers[self._reader].slot == self._layers[layer].slot
        ):
            raise RuntimeError(
                "Cannot overwrite a weight slot before releasing its reader"
            )
        self._stream.wait_event(slot.released)
        for destination, source, size in self._plans[layer]:
            vmm.copy(destination, source, size, self._stream)
        slot.ready.record(self._stream)
        slot.layer = layer

    def acquire(self, layer: int) -> None:
        self._require_ready()
        if self._reader is not None:
            raise RuntimeError("The previous weight reader must be released first")
        self.prefetch(layer)
        stream = torch.cuda.current_stream(self.device)
        slot = self._slots[self._layers[layer].slot]
        if slot.last_reader_stream != stream.cuda_stream:
            stream.wait_event(slot.released)
        stream.wait_event(slot.ready)
        self._reader = layer
        self._reader_stream = stream.cuda_stream

    def release(self, layer: int) -> None:
        self._require_ready()
        stream = torch.cuda.current_stream(self.device)
        if self._reader != layer or self._reader_stream != stream.cuda_stream:
            raise RuntimeError(
                "Release must follow the matching reader on its compute stream"
            )
        slot = self._slots[self._layers[layer].slot]
        slot.released.record(stream)
        slot.last_reader_stream = stream.cuda_stream
        self._reader = None
        self._reader_stream = None

    def close(self) -> None:
        if self._closed:
            return
        if self._reader is not None:
            raise RuntimeError(
                "Release the final weight reader before closing the store"
            )
        self._published.synchronize()
        self._stream.synchronize()
        for slot in self._slots:
            slot.released.synchronize()
        if self._finalized:
            dist.barrier(group=self.group)
        self._views.clear()
        for address, size in reversed(self._mappings):
            vmm.unmap(address, size)
        for address, size in reversed(self._reservations):
            vmm.free_address(address, size)
        for handle in reversed(self._handles):
            vmm.release_handle(handle)
        self._closed = True
