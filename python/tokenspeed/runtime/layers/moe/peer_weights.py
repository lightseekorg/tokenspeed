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

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from tokenspeed_kernel.ops.communication.peer_weights import (
    ExpertPartition,
    PeerWeightStore,
    PreparedExpertWeights,
)

from tokenspeed.runtime.execution.breakable_cuda_graph import run_eager

logger = logging.getLogger(__name__)


@runtime_checkable
class PeerWeightOwner(Protocol):
    def close_peer_weights(self) -> None: ...

    def peer_weights_controller(self) -> MoEPeerWeights | None: ...


def peer_weights_controller_for(model: object) -> MoEPeerWeights | None:
    if not isinstance(model, PeerWeightOwner):
        return None
    controller = model.peer_weights_controller()
    return controller if isinstance(controller, MoEPeerWeights) else None


@dataclass
class _LayerWeights:
    module: torch.nn.Module
    names: tuple[str, ...]
    next_layer: int | None


class MoEPeerWeights:
    """Model-owned orchestration of sharded storage and full-expert compute.

    Publish processed layers incrementally to release their old allocations.
    Finalize before tuning or cache sizing; forward calls remain on the data
    plane. The caller joins any side-stream weight readers before release.
    """

    def __init__(self, *, partition: ExpertPartition, store: PeerWeightStore) -> None:
        self.partition = partition
        self.store = store
        self._layers: dict[int, _LayerWeights] = {}
        self._first_layer: int | None = None
        self._finalized = False

    def publish(
        self,
        layer: int,
        module: torch.nn.Module,
        prepared: PreparedExpertWeights,
    ) -> None:
        if self._finalized or layer in self._layers:
            raise RuntimeError(
                f"Cannot republish immutable expert weights for layer {layer}"
            )
        owned = self.store.publish_layer(layer, prepared.shards, layout=prepared.layout)
        prepared.bind(module, owned)
        self._layers[layer] = _LayerWeights(module, tuple(owned), None)

    def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("Peer weights are already finalized")
        self.store.finalize()
        order = list(self._layers)
        self._first_layer = order[0]
        for index, layer in enumerate(order):
            binding = self._layers[layer]
            views = self.store.weights(layer)
            for name in binding.names:
                binding.module.get_parameter(name).data = views[name]
            binding.next_layer = order[index + 1] if index + 1 < len(order) else None
        self._finalized = True
        logger.info(
            f"DWDP weights ready: owners={self.partition.size}, "
            f"owned_experts={self.partition.count}/{self.partition.num_experts}, "
            f"layers={len(order)}, physical_bytes={self.store.physical_bytes}, "
            f"staging_bytes={self.store.staging_bytes}"
        )

    def prefetch_first(self) -> None:
        if self._first_layer is None:
            raise RuntimeError("Finalize peer weights before execution")
        run_eager(lambda: self.store.prefetch(self._first_layer))

    def acquire(self, layer: int) -> None:
        # Stable views are bound at finalize; only the stream ordering and
        # host-side staging state must run again between graph segments.
        run_eager(lambda: self.store.acquire(layer))

    def release(self, layer: int) -> None:
        def release_and_prefetch() -> None:
            self.store.release(layer)
            next_layer = self._layers[layer].next_layer
            if next_layer is not None:
                self.store.prefetch(next_layer)

        run_eager(release_and_prefetch)

    @contextmanager
    def read(self, layer: int):
        self.acquire(layer)
        try:
            yield
        finally:
            self.release(layer)

    @property
    def finalized(self) -> bool:
        """Whether ``finalize()`` has bound full-expert peer views."""
        return self._finalized

    @property
    def released(self) -> bool:
        """Whether the underlying peer-weight physical storage is currently released."""
        return self.store.released

    def managed_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """Return every registered layer parameter bound to this controller in layer order."""
        return tuple(
            binding.module.get_parameter(name)
            for binding in self._layers.values()
            for name in binding.names
        )

    def vmm_managed_storages(self) -> frozenset[tuple[int, int]]:
        """Return ``(data_ptr, nbytes)`` pairs for VMM-backed peer tensors after finalization.

        Raises ``RuntimeError`` if called before ``finalize()`` or if VMM tensors exist
        without any allocated physical memory on the store.
        """
        if not self._finalized:
            raise RuntimeError("Finalize peer weights before inspecting storage")
        storages = frozenset(
            (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            for tensor in self.store.vmm_managed_tensors()
            if tensor.numel()
        )
        if storages and self.store.physical_bytes <= 0:
            raise RuntimeError("Finalized peer weight store reports no physical memory")
        return storages

    def owned_weights(self, layer: int) -> dict[str, torch.Tensor]:
        """Return this rank's writable local shard tensors for ``layer`` after finalization."""
        if not self._finalized:
            raise RuntimeError("Finalize peer weights before accessing owned shards")
        return self.store.owned_weights(layer)

    def prepare_weight_update(self) -> None:
        """Quiesce peer readers and copy streams before rewriting owned shards; requires finalization."""
        if not self._finalized:
            raise RuntimeError("Finalize peer weights before updating weights")
        self.store.prepare_weight_update()

    def refresh_weights(self) -> None:
        """Refresh boundary intervals and invalidate staging after owned shards change; requires finalization."""
        if not self._finalized:
            raise RuntimeError("Finalize peer weights before refreshing weights")
        self.store.refresh_weights()

    def release_storage(self, *, backup: bool) -> None:
        """Release VMM physical storage after finalization, backing up owned shards to CPU when ``backup`` is True."""
        if not self._finalized:
            raise RuntimeError("Finalize peer weights before releasing storage")
        self.store.release_storage(backup=backup)

    def resume_storage(self, *, backup: bool) -> None:
        """Reallocate and remap VMM storage after release, restoring owned shards from CPU when ``backup`` is True."""
        if not self._finalized:
            raise RuntimeError("Finalize peer weights before resuming storage")
        self.store.resume_storage(backup=backup)

    def close(self) -> None:
        self.store.close()
