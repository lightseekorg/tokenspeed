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
        self.slots = [torch.empty(4, 3), torch.empty(4, 3)]
        self.resident: list[int | None] = [None, None]
        self.reader: int | None = None

    def publish_layer(self, layer, shards, *, layout):
        assert layout == "cpu-test"
        (shard,) = shards
        owned = shard.tensor.clone()
        self.sources[layer] = torch.cat((owned, owned + 10))
        return {shard.name: owned}

    def finalize(self):
        pass

    def weights(self, layer):
        return {"weight": self.slots[layer % 2]}

    def prefetch(self, layer):
        slot = layer % 2
        assert self.reader is None or self.reader % 2 != slot
        self.slots[slot].copy_(self.sources[layer])
        self.resident[slot] = layer

    def acquire(self, layer):
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
        return sum(t.numel() * t.element_size() for t in self.sources.values())

    @property
    def staging_bytes(self):
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
