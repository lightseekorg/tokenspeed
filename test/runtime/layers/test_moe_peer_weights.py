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

import pytest
import torch
from tokenspeed_kernel.ops.communication.peer_weights import (
    ExpertPartition,
    ExpertWeightShard,
    PreparedExpertWeights,
)

from tokenspeed.runtime.layers.moe.peer_weights import MoEPeerWeights


class _CpuStore:
    """Replace GPU peer copies with CPU copies into the same two staging slots."""

    def __init__(self):
        self.sources: dict[int, torch.Tensor] = {}
        self.owned: dict[int, torch.Tensor] = {}
        self.backups: dict[int, torch.Tensor] = {}
        self.slots = [torch.empty(4, 3), torch.empty(4, 3)]
        self.resident: list[int | None] = [None, None]
        self.reader: int | None = None
        self._released = False
        self.events: list[str] = []

    def publish_layer(self, layer, shards, *, layout):
        assert layout == "cpu-test"
        (shard,) = shards
        owned = shard.tensor.clone()
        self.owned[layer] = owned
        self.sources[layer] = torch.cat((owned, owned + 10))
        return {shard.name: owned}

    def finalize(self):
        pass

    def weights(self, layer):
        return {"weight": self.slots[layer % 2]}

    def owned_weights(self, layer):
        assert not self._released
        return {"weight": self.owned[layer]}

    def vmm_managed_tensors(self):
        return tuple(self.slots[layer % 2] for layer in self.sources)

    def prepare_weight_update(self):
        assert not self._released and self.reader is None
        self.resident = [None, None]
        self.events.append("prepare")

    def refresh_weights(self):
        assert not self._released and self.reader is None
        for layer, owned in self.owned.items():
            self.sources[layer] = torch.cat((owned, owned + 10))
        self.resident = [None, None]
        self.events.append("refresh")

    def release_storage(self, *, backup):
        assert not self._released and self.reader is None
        if backup:
            self.backups = {layer: t.clone() for layer, t in self.owned.items()}
        for t in self.owned.values():
            t.fill_(float("nan"))
        self.resident = [None, None]
        self._released = True
        self.events.append(f"release:{backup}")

    def resume_storage(self, *, backup):
        assert self._released
        if backup:
            for layer, t in self.owned.items():
                t.copy_(self.backups[layer])
            self.backups.clear()
            for layer, owned in self.owned.items():
                self.sources[layer] = torch.cat((owned, owned + 10))
        self.resident = [None, None]
        self._released = False
        self.events.append(f"resume:{backup}")

    @property
    def released(self):
        return self._released

    def prefetch(self, layer):
        assert not self._released
        slot = layer % 2
        assert self.reader is None or self.reader % 2 != slot
        self.slots[slot].copy_(self.sources[layer])
        self.resident[slot] = layer

    def acquire(self, layer):
        assert not self._released
        assert self.reader is None
        assert self.resident[layer % 2] == layer
        self.reader = layer

    def release(self, layer):
        assert self.reader == layer
        self.reader = None

    def close(self):
        assert self.reader is None

    @property
    def physical_bytes(self):
        if self._released:
            return 0
        return sum(t.numel() * t.element_size() for t in self.sources.values())

    @property
    def staging_bytes(self):
        if self._released:
            return 0
        return sum(t.numel() * t.element_size() for t in self.slots)


def _manager(num_layers):
    store = _CpuStore()
    manager = MoEPeerWeights(partition=ExpertPartition(4, 0, 2), store=store)
    layers = []
    for index in range(num_layers):
        layer = torch.nn.Linear(3, 2, bias=False)
        layer.weight.data.fill_(index + 1)
        manager.publish(
            index,
            layer,
            PreparedExpertWeights(
                "cpu-test", (ExpertWeightShard("weight", layer.weight, False),)
            ),
        )
        layers.append(layer)
    manager.finalize()
    return manager, layers, store


@pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
def test_stable_full_expert_views_across_queued_forwards(num_layers):
    manager, layers, store = _manager(num_layers)
    addresses = [layer.weight.data_ptr() for layer in layers]
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    for _ in range(5):
        manager.prefetch_first()
        for index, layer in enumerate(layers):
            with manager.read(index):
                torch.testing.assert_close(
                    layer(inputs), inputs @ store.sources[index].T, rtol=0, atol=0
                )
        assert [layer.weight.data_ptr() for layer in layers] == addresses
    manager.close()


@pytest.mark.parametrize("failed_layer", [0, 1, 2])
def test_reader_exception_releases_before_another_forward(failed_layer):
    manager, layers, store = _manager(3)
    manager.prefetch_first()
    for index in range(failed_layer):
        with manager.read(index):
            pass
    with pytest.raises(ValueError, match="reader failed"):
        with manager.read(failed_layer):
            raise ValueError("reader failed")
    manager.prefetch_first()
    for index, layer in enumerate(layers):
        with manager.read(index):
            torch.testing.assert_close(layer.weight, store.sources[index])
    manager.close()


def test_peer_weight_update_and_sleep_wake_preserve_view_addresses():
    from tokenspeed.runtime.model_loader.weight_utils import initialize_dummy_weights
    from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
    from tokenspeed.runtime.models.base.weight_update import weight_update_session

    manager, layers, store = _manager(2)
    addresses = [layer.weight.data_ptr() for layer in layers]
    assert len(manager.vmm_managed_storages()) == 2

    class _PeerModel(BaseCausalLM):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.dense = torch.nn.Linear(3, 3, bias=False)
            self.layers = torch.nn.ModuleList(layers)
            self.events: list[str] = []

        def peer_weights_controller(self):
            return manager

        def close_peer_weights(self):
            manager.close()

        def load_weights(self, weights):
            self.events.append("load")
            for name, value in weights:
                manager.owned_weights(int(name))["weight"].fill_(float(value.item()))
            self.post_load_weights()

        def post_load_weights(self):
            self.events.append("post_load")

    model = _PeerModel()
    before_peer = [layer.weight.clone() for layer in layers]
    before_dense = model.dense.weight.clone()
    initialize_dummy_weights(
        model,
        exclude_data_ptrs={
            p.data_ptr() for p in manager.managed_parameters() if p.numel()
        },
    )
    assert not torch.equal(model.dense.weight, before_dense)
    for layer, expected in zip(layers, before_peer, strict=True):
        assert torch.equal(layer.weight, expected)

    # Preflight failure aborts before quiescing peer storage.
    with pytest.raises(ValueError, match="bad checkpoint"):
        with weight_update_session(
            [model],
            require_complete=False,
            cpu_group=None,
            preflight=lambda: (_ for _ in ()).throw(ValueError("bad checkpoint")),
        ):
            raise AssertionError("body must not run after preflight failure")
    assert store.events == []
    assert not model._weight_update_active

    # Failure in prepare_weight_update aborts opened models so later updates still work.
    real_prepare = store.prepare_weight_update

    def _fail_prepare() -> None:
        raise RuntimeError("peer quiesce failed")

    store.prepare_weight_update = _fail_prepare
    try:
        with pytest.raises(RuntimeError, match="peer quiesce failed"):
            with weight_update_session([model], require_complete=False, cpu_group=None):
                raise AssertionError("body must not run after prepare failure")
    finally:
        store.prepare_weight_update = real_prepare
    assert not model._weight_update_active

    # Resident live update refreshes owned shards and derived state in order.
    with weight_update_session([model], require_complete=False, cpu_group=None):
        model.load_weights([("0", torch.tensor(21.0)), ("1", torch.tensor(22.0))])
    assert store.events == ["prepare", "refresh"]
    assert model.events == ["load", "post_load"]
    assert [layer.weight.data_ptr() for layer in layers] == addresses

    # CPU-backup sleep/wake preserves weights and addresses.
    manager.release_storage(backup=True)
    assert manager.released and store.physical_bytes == 0
    manager.resume_storage(backup=True)
    assert not manager.released and store.physical_bytes > 0
    manager.prefetch_first()
    for index, layer in enumerate(layers):
        with manager.read(index):
            torch.testing.assert_close(layer.weight, store.sources[index])

    # Discard sleep/wake requires reload to restore valid contents.
    manager.release_storage(backup=False)
    manager.resume_storage(backup=False)
    with weight_update_session([model], require_complete=False, cpu_group=None):
        model.load_weights([("0", torch.tensor(31.0)), ("1", torch.tensor(32.0))])
    manager.prefetch_first()
    for index, layer in enumerate(layers):
        with manager.read(index):
            assert float(layer.weight[0, 0]) == 31.0 + index
            assert float(layer.weight[2, 0]) == 41.0 + index
    assert [layer.weight.data_ptr() for layer in layers] == addresses
    manager.close()
