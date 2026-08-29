from types import SimpleNamespace
from unittest.mock import Mock

import torch

from tokenspeed.runtime.distributed.comm_backend import hccl as hccl_module
from tokenspeed.runtime.distributed.comm_backend import registry
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend


def test_registry_selects_hccl_on_npu(monkeypatch):
    monkeypatch.setattr(registry, "_global_backend", None)
    monkeypatch.setattr(
        registry,
        "current_platform",
        lambda: SimpleNamespace(is_npu=True),
    )

    backend = registry.initialize_comm_backend()

    assert isinstance(backend, HcclBackend)


def test_hccl_all_reduce_uses_hccl_process_group(monkeypatch):
    process_group = Mock()
    get_process_group = Mock(return_value=process_group)
    all_reduce = Mock()
    monkeypatch.setattr(hccl_module.pg_manager, "get_process_group", get_process_group)
    monkeypatch.setattr(hccl_module.dist, "all_reduce", all_reduce)
    tensor = torch.ones(4)

    output = HcclBackend().all_reduce(tensor, (0, 1))

    assert output is tensor
    get_process_group.assert_called_once_with("hccl", (0, 1))
    all_reduce.assert_called_once_with(
        tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )
