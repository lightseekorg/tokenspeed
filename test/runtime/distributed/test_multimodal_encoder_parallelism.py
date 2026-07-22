# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT

"""Coverage for multimodal encoder weight-TP and item-DP execution."""

from __future__ import annotations

import argparse
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.multimodal.embedder import (
    EncodePlan,
    EncoderSpec,
    MultimodalEmbedder,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def _mapping_from_cli(
    mode: str | None = None, *, request_dp_size: int = 1, rank: int = 0
) -> Mapping:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = ["--model", "test/model", "--tensor-parallel-size", "2"]
    if mode is not None:
        argv.extend(["--mm-encoder-tp-mode", mode])
    if request_dp_size > 1:
        argv.extend(["--data-parallel-size", str(request_dp_size)])
    args = parser.parse_args(argv)
    with patch.object(ServerArgs, "__post_init__"):
        server_args = ServerArgs.from_cli_args(args)
    server_args.resolve_basic_defaults()
    server_args.resolve_parallelism()
    server_args.mapping.rank = rank
    return server_args.mapping


def _item(item_id: int, rows: int) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=item_id,
        offsets=[(0, rows - 1)],
        feature=torch.tensor([item_id]),
    )


def _encoded_rows(
    item: MultimodalDataItem, width: int, device: torch.device
) -> torch.Tensor:
    rows = sum(end - start + 1 for start, end in item.offsets)
    row_ids = torch.arange(rows, dtype=torch.float32, device=device)
    item_ids = torch.full_like(row_ids, float(item.hash))
    columns = [item_ids, row_ids]
    if width == 4:
        columns.extend([item_ids + 1000, row_ids + 1000])
    return torch.stack(columns, dim=1)


def test_multimodal_encoder_weight_tp() -> None:
    mapping = _mapping_from_cli(rank=1)
    assert (mapping.vision.tp_size, mapping.vision.dp_size) == (2, 1)
    assert mapping.vision.tp_group == mapping.attn.tp_group == (0, 1)

    items = [_item(10, 2), _item(20, 3)]
    calls: list[list[int]] = []

    def encoder(batch):
        calls.append([int(item.hash) for item in batch])
        return torch.cat(
            [_encoded_rows(item, 2, torch.device("cpu")) for item in batch], dim=0
        )

    embedder = MultimodalEmbedder(encoder_mapping=mapping.vision)
    embedder._encode(
        EncodePlan(misses_by_modality={Modality.IMAGE: items}),
        {Modality.IMAGE: EncoderSpec(encoder)},
        SimpleNamespace(),
        torch.device("cpu"),
        2,
        torch.float32,
    )

    assert calls == [[10, 20]]
    for item in items:
        torch.testing.assert_close(
            item.encoded, _encoded_rows(item, 2, torch.device("cpu"))
        )


def _run_item_dp_case(rank: int, device: torch.device, mapping: Mapping) -> None:
    items = [_item(10, 1), _item(20, 4), _item(30, 2)]
    owned_hashes_by_rank = ([20], [10, 30])
    calls: list[list[int]] = []

    def encoder(batch):
        calls.append([int(item.hash) for item in batch])
        assert all(item.feature.device == device for item in batch)
        return torch.cat([_encoded_rows(item, 4, device) for item in batch], dim=0)

    model = SimpleNamespace(
        deepstack_visual_indexes=[0],
        separate_deepstack_embeds=lambda output: (output[:, :2], output[:, 2:]),
    )
    embedder = MultimodalEmbedder(encoder_mapping=mapping.vision)
    embedder._encode(
        EncodePlan(misses_by_modality={Modality.IMAGE: items}),
        {
            Modality.IMAGE: EncoderSpec(
                encoder,
                deepstack=True,
            )
        },
        model,
        device,
        2,
        torch.float32,
    )

    assert calls == [owned_hashes_by_rank[rank]]
    for item in items:
        expected = _encoded_rows(item, 4, device)
        torch.testing.assert_close(item.encoded, expected[:, :2])
        torch.testing.assert_close(item.encoded_deepstack, expected[:, 2:])

    # One item leaves rank 1 idle; it must skip the encoder and still receive
    # the exact-size output from rank 0.
    idle_item = _item(40, 3)
    idle_calls = 0

    def idle_encoder(batch):
        nonlocal idle_calls
        idle_calls += 1
        return _encoded_rows(batch[0], 2, device)

    embedder._encode(
        EncodePlan(misses_by_modality={Modality.IMAGE: [idle_item]}),
        {Modality.IMAGE: EncoderSpec(idle_encoder)},
        SimpleNamespace(),
        device,
        2,
        torch.float32,
    )
    assert idle_calls == (1 if rank == 0 else 0)
    torch.testing.assert_close(idle_item.encoded, _encoded_rows(idle_item, 2, device))


def _item_dp_worker(rank: int, world_size: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager,
        )

        mapping = Mapping(
            rank=rank,
            world_size=world_size,
            attn_tp_size=world_size,
            vision_tp_size=1,
            vision_dp_size=world_size,
        )
        process_group_manager.init_process_group(
            mapping.vision.dp_group, backend="nccl"
        )
        _run_item_dp_case(rank, device, mapping)
    finally:
        dist.destroy_process_group()


def test_multimodal_encoder_item_dp() -> None:
    # Item-DP is scoped to each attention TP group, independently of outer
    # request DP. Rank 3 belongs to the second request-DP replica.
    mapping = _mapping_from_cli("data", request_dp_size=2, rank=3)
    assert (mapping.vision.tp_size, mapping.vision.dp_size) == (1, 2)
    assert mapping.vision.dp_group == (2, 3)

    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _item_dp_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )
