from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager
from tokenspeed.runtime.execution import distributed_initializer


def _mapping(world_size: int = 4) -> Mapping:
    return Mapping(
        rank=0,
        world_size=world_size,
        attn_tp_size=world_size,
        dense_tp_size=world_size,
        moe_tp_size=world_size,
    )


def test_nccl_device_id_is_used_for_root_and_child_groups(monkeypatch):
    mapping = _mapping()
    device_id = torch.device("cuda", 4)
    init_process_group = Mock()
    new_group = Mock(side_effect=lambda ranks, **kwargs: (ranks, kwargs))
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.init_process_group",
        init_process_group,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.get_world_size",
        lambda: mapping.world_size,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.new_group",
        new_group,
    )

    manager = ProcessGroupManager()
    manager.init_distributed(
        mapping,
        distributed_init_method="tcp://127.0.0.1:12345",
        backend="nccl",
        timeout=30,
        device_id=device_id,
    )
    manager.init_process_group((0, 1), backend=["nccl", "gloo"])

    init_process_group.assert_called_once_with(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=4,
        rank=0,
        timeout=timedelta(seconds=30),
        device_id=device_id,
    )
    assert new_group.call_args_list == [
        call((0, 1), backend="nccl", device_id=device_id),
        call((2, 3), backend="nccl", device_id=device_id),
        call((0, 1), backend="gloo"),
        call((2, 3), backend="gloo"),
    ]


def test_non_nccl_groups_do_not_receive_device_id(monkeypatch):
    mapping = _mapping(world_size=1)
    init_process_group = Mock()
    new_group = Mock(return_value=object())
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.init_process_group",
        init_process_group,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.get_world_size",
        lambda: mapping.world_size,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.dist.new_group",
        new_group,
    )

    manager = ProcessGroupManager()
    manager.init_distributed(
        mapping,
        backend="gloo",
        device_id=torch.device("cuda", 4),
    )
    manager.init_process_group(mapping.world_group, backend="gloo")

    init_process_group.assert_called_once_with(
        backend="gloo",
        init_method="env://",
        world_size=1,
        rank=0,
        timeout=None,
    )
    new_group.assert_called_once_with((0,), backend="gloo")


def test_distributed_initializer_uses_physical_cuda_device(monkeypatch):
    mapping = _mapping(world_size=1)
    device_module = SimpleNamespace(set_device=Mock())
    pg_manager = Mock()
    pg_manager.get_process_group.return_value = object()
    monkeypatch.setattr(
        distributed_initializer.torch,
        "get_device_module",
        lambda device: device_module,
    )
    monkeypatch.setattr(
        distributed_initializer, "get_available_gpu_memory", lambda *args, **kwargs: 80
    )
    monkeypatch.setattr(
        distributed_initializer, "maybe_set_numa_aware_cpu_affinity", Mock()
    )
    monkeypatch.setattr(distributed_initializer, "pg_manager", pg_manager)
    config = distributed_initializer.DistributedConfig(
        device="cuda",
        gpu_id=4,
        world_size=1,
        global_rank=0,
        local_rank=0,
        attn_tp_rank=0,
        attn_tp_size=1,
        dp_size=1,
        dense_tp_size=1,
        moe_ep_size=1,
        moe_ep_rank=0,
        nccl_port=12345,
        mapping=mapping,
    )

    assert distributed_initializer.DistributedInitializer.initialize(config) == 80

    device_module.set_device.assert_called_once_with(4)
    pg_manager.init_distributed.assert_called_once_with(
        mapping,
        backend="nccl",
        distributed_init_method="tcp://127.0.0.1:12345",
        timeout=1800,
        device_id=torch.device("cuda", 4),
    )
