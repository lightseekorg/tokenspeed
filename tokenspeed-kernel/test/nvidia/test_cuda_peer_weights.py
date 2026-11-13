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

"""Run with torchrun --nproc-per-node=2 -m pytest <this file> on a fabric host."""
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.peer_weights import (
    ExpertPartition,
    ExpertWeightShard,
    create_peer_weight_store,
)


@pytest.fixture(scope="module")
def peers():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Launch under torchrun with at least two fabric-connected GPUs")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    partition = ExpertPartition(
        dist.get_world_size() * 2, dist.get_rank(), dist.get_world_size()
    )
    yield partition, device
    dist.destroy_process_group()


def _store(partition, device):
    return create_peer_weight_store(
        partition=partition,
        group=dist.group.WORLD,
        device=device,
        model_identity="peer-weights-test",
        pool_page_bytes=16 * 1024**2,
        copy_implementation="simple",
        solution="cuda",
    )


def _values(first, count, width, layer, dtype, device):
    ids = torch.arange(first, first + count, device=device).reshape(-1, 1)
    columns = torch.arange(width, device=device).reshape(1, -1)
    return ((ids * 17 + columns + layer * 13) % 113).to(dtype)


@pytest.mark.parametrize("layers", [2, 3])
def test_composite_weights_and_queued_forward_lifetime(peers, layers):
    partition, device = peers
    store = _store(partition, device)
    fields = (
        ("weight", 1048585, torch.uint8),
        ("scale", 524291, torch.float32),
        ("fp8", 131077, torch.float8_e4m3fn),
    )
    try:
        for layer in range(layers):
            shards = tuple(
                ExpertWeightShard(
                    name,
                    _values(
                        partition.first, partition.count, width, layer, dtype, device
                    ),
                    False,
                )
                for name, width, dtype in fields
            )
            shards += (
                ExpertWeightShard(
                    "alpha",
                    _values(
                        partition.first,
                        partition.count,
                        1,
                        layer,
                        torch.float32,
                        device,
                    ),
                    True,
                ),
            )
            owned = store.publish_layer(layer, shards, layout="expert-major-test")
            for shard in shards:
                torch.testing.assert_close(
                    owned[shard.name].float(), shard.tensor.float(), rtol=0, atol=0
                )
        store.finalize()
        stable = {
            layer: {
                name: value.data_ptr() for name, value in store.weights(layer).items()
            }
            for layer in range(layers)
        }
        for layer in range(layers):
            store.acquire(layer)
            for name, width, dtype in (*fields, ("alpha", 1, torch.float32)):
                expected = _values(
                    0, partition.num_experts, width, layer, dtype, device
                )
                torch.testing.assert_close(
                    store.weights(layer)[name].float(), expected.float(), rtol=0, atol=0
                )
            store.release(layer)
        checks = []
        expected_checks = []
        for iteration in range(128):
            for layer in range(layers):
                store.acquire(layer)
                torch.cuda._sleep(100000)
                value = store.weights(layer)["weight"]
                sample = value[:, ::4093].sum(dtype=torch.int64)
                checks.append(sample)
                expected_checks.append(layer)
                store.release(layer)
                if layer + 1 < layers:
                    store.prefetch(layer + 1)
        torch.cuda.synchronize()
        references = [
            int(
                _values(
                    0, partition.num_experts, fields[0][1], layer, torch.uint8, device
                )[:, ::4093]
                .sum()
                .item()
            )
            for layer in range(layers)
        ]
        assert [int(x.item()) for x in checks] == [
            references[x] for x in expected_checks
        ]
        assert stable == {
            layer: {
                name: value.data_ptr() for name, value in store.weights(layer).items()
            }
            for layer in range(layers)
        }
        full_layer_bytes = sum(
            partition.num_experts * width * torch.empty((), dtype=dtype).element_size()
            for _, width, dtype in fields
        )
        two_slot_bound = 2 * (full_layer_bytes + 2 * len(fields) * 16 * 1024**2)
        assert 0 < store.staging_bytes <= two_slot_bound
        # Every rank can pull while all other owners are idle in the CPU barrier.
        for active in range(partition.size):
            if partition.rank == active:
                for layer in reversed(range(layers)):
                    store.acquire(layer)
                    torch.testing.assert_close(
                        store.weights(layer)["weight"],
                        _values(
                            0,
                            partition.num_experts,
                            fields[0][1],
                            layer,
                            torch.uint8,
                            device,
                        ),
                        rtol=0,
                        atol=0,
                    )
                    store.release(layer)
            dist.barrier()
    finally:
        store.close()
    store.close()
    with pytest.raises(RuntimeError, match="not available"):
        store.acquire(0)


def test_reader_must_be_released_before_slot_reuse(peers):
    partition, device = peers
    store = _store(partition, device)
    try:
        for layer in range(3):
            store.publish_layer(
                layer,
                (
                    ExpertWeightShard(
                        "weight",
                        torch.ones((partition.count, 1025), device=device),
                        False,
                    ),
                ),
                layout="test",
            )
        store.finalize()
        store.acquire(0)
        with pytest.raises(RuntimeError, match="before releasing"):
            store.prefetch(2)
        with pytest.raises(RuntimeError, match="previous weight reader"):
            store.acquire(1)
        with pytest.raises(RuntimeError, match="matching reader"):
            store.release(1)
        with pytest.raises(RuntimeError, match="final weight reader"):
            store.close()
        store.release(0)
        store.acquire(2)
        store.release(2)
    finally:
        store.close()


def test_full_model_mapping_count(peers):
    partition, device = peers
    store = _store(partition, device)
    try:
        for layer in range(92):
            store.publish_layer(
                layer,
                tuple(
                    ExpertWeightShard(
                        str(field),
                        _values(
                            partition.first,
                            partition.count,
                            1025 + field,
                            layer,
                            torch.uint8,
                            device,
                        ),
                        False,
                    )
                    for field in range(4)
                ),
                layout="four-fields-test",
            )
        store.finalize()
        for layer in range(92):
            store.acquire(layer)
            for field in range(4):
                torch.testing.assert_close(
                    store.weights(layer)[str(field)],
                    _values(
                        0,
                        partition.num_experts,
                        1025 + field,
                        layer,
                        torch.uint8,
                        device,
                    ),
                    rtol=0,
                    atol=0,
                )
            store.release(layer)
    finally:
        store.close()


@pytest.mark.parametrize("switch_stream", [False, True])
def test_slot_reuse_waits_for_readers_across_streams(peers, switch_stream):
    partition, device = peers
    store = _store(partition, device)
    width = 1048585
    try:
        for layer in range(3):
            store.publish_layer(
                layer,
                (
                    ExpertWeightShard(
                        "weight",
                        torch.full(
                            (partition.count, width), float(layer + 1), device=device
                        ),
                        False,
                    ),
                ),
                layout="stream-order-test",
            )
        store.finalize()
        assert store.staging_bytes > 0
        slow = torch.cuda.Stream(device=device)
        fast = torch.cuda.Stream(device=device) if switch_stream else slow
        delayed = torch.empty((partition.num_experts, 2), device=device)
        immediate = torch.empty_like(delayed)
        overwritten = torch.empty_like(delayed)
        # Warm copies before the race; allocation synchronization can hide it.
        store.acquire(0)
        for output in (delayed, immediate, overwritten):
            output.copy_(store.weights(0)["weight"][:, :: width - 1])
        store.release(0)
        torch.cuda.synchronize()
        with torch.cuda.stream(slow):
            store.acquire(0)
            torch.cuda._sleep(300000000)
            delayed.copy_(store.weights(0)["weight"][:, :: width - 1])
            store.release(0)
        with torch.cuda.stream(fast):
            store.acquire(0)
            immediate.copy_(store.weights(0)["weight"][:, :: width - 1])
            store.release(0)
        store.acquire(2)
        overwritten.copy_(store.weights(2)["weight"][:, :: width - 1])
        store.release(2)
        torch.cuda.synchronize()
        torch.testing.assert_close(delayed, torch.ones_like(delayed), rtol=0, atol=0)
        torch.testing.assert_close(
            immediate, torch.ones_like(immediate), rtol=0, atol=0
        )
        torch.testing.assert_close(
            overwritten, torch.full_like(overwritten, 3), rtol=0, atol=0
        )
    finally:
        store.close()
