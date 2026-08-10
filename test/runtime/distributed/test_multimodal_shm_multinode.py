# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT

"""Two-node SHM transport coverage, launched once per node with torchrun."""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import VisionTowerMapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager,
)
from tokenspeed.runtime.multimodal.encoder_feature_transport import (
    EncoderFeatureTransport,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.multimodal.shm_transport import (
    ShmTensorHandle,
    prepare_shm_features,
)


def _request(items: list[MultimodalDataItem]):
    return SimpleNamespace(multimodal_inputs=MultimodalInputs(mm_items=items))


def _publish_items(rank: int) -> tuple[list[MultimodalDataItem], list[torch.Tensor]]:
    expected = [
        torch.tensor([3, 1, 4, 1, 5], dtype=torch.uint8),
        torch.tensor([2, 7, 1], dtype=torch.uint8),
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
    ]
    metadata = None
    handles = None
    if rank == 0:
        handles = [ShmTensorHandle.publish(tensor) for tensor in expected]
        metadata = [(handle.shm_name, handle.shape, handle.dtype) for handle in handles]

    objects = [metadata]
    dist.broadcast_object_list(objects, src=0)
    if handles is None:
        handles = [
            ShmTensorHandle(name, tuple(shape), dtype)
            for name, shape, dtype in objects[0]
        ]
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=index,
            offsets=[(0, 0)],
            feature_shm=handle,
        )
        for index, handle in enumerate(handles)
    ]
    return items, expected


def _release(items: list[MultimodalDataItem]) -> None:
    for item in items:
        if item.feature_shm is not None:
            item.feature_shm.release()
            item.feature_shm = None


def test_multinode_owner_and_tp_transport() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch this test with torchrun --nnodes=2")
    if not torch.cuda.is_available():
        pytest.skip("multinode transport coverage requires CUDA")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    process_group_manager.register_process_group("gloo", (0, 1), dist.group.WORLD)
    nccl_group = None
    owner_items: list[MultimodalDataItem] = []
    tp_items: list[MultimodalDataItem] = []
    try:
        hostnames: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(hostnames, socket.gethostname())
        if len(set(hostnames)) != 2:
            pytest.skip("multinode transport coverage requires two distinct hosts")

        owner_items, expected = _publish_items(rank)
        prepare_shm_features([_request(owner_items)], dist.group.WORLD)
        assert all(item.feature_shm.is_cross_node for item in owner_items)
        owner_transport = EncoderFeatureTransport(
            VisionTowerMapping(rank=rank, world_size=2, tp_size=1, dp_size=2)
        )
        owner_transport.route_to_item_owners(
            owner_items,
            ((0,), (1, 2)),
        )
        owned_indices = (0,) if rank == 0 else (1, 2)
        for index in owned_indices:
            received = owner_items[index].feature_shm.consume()
            assert received.is_pinned()
            torch.testing.assert_close(received, expected[index])
        dist.barrier()
        _release(owner_items)
        dist.barrier()

        tp_items, expected = _publish_items(rank)
        prepare_shm_features([_request(tp_items)], dist.group.WORLD)
        assert all(item.feature_shm.is_cross_node for item in tp_items)

        tp_group = (0, 1)
        nccl_group = dist.new_group(ranks=list(tp_group), backend="nccl")
        process_group_manager.register_process_group("nccl", tp_group, nccl_group)
        mapping = VisionTowerMapping(rank=rank, world_size=2, tp_size=2, dp_size=1)
        transport = EncoderFeatureTransport(mapping)
        device = torch.device("cuda", local_rank)
        transport.move_to_device(tp_items, device)

        for item, tensor in zip(tp_items, expected, strict=True):
            assert item.feature_shm is None
            assert item.feature is not None
            torch.testing.assert_close(item.feature.cpu(), tensor)
        dist.barrier()
    finally:
        _release(owner_items)
        _release(tp_items)
        if nccl_group is not None:
            dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()
