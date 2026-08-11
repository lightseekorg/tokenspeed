from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.distributed.comm_backend import (
    triton_allreduce as triton_allreduce_module,
)
from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
from tokenspeed.runtime.distributed.comm_backend.triton_allreduce import (
    TritonAllReduceBackend,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture
def backend(monkeypatch):
    instance = AutoBackend()
    monkeypatch.setattr(instance, "_nccl", Mock())
    monkeypatch.setattr(instance, "_rsag", Mock())
    monkeypatch.setattr(instance, "_trtllm_ar", Mock())
    monkeypatch.setattr(instance, "_triton_ar", Mock())
    instance._triton_ar.producer_direct_max_bytes = 1024 * 1024
    instance._triton_ar.can_plan_all_reduce.return_value = True
    return instance


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("token_all_gather", (torch.empty(1, 4), (0, 1), [1, 1])),
        ("token_reduce_scatter", (torch.empty(2, 4), (0, 1), [1, 1])),
        ("all_gather", (torch.empty(1, 4), (0, 1), -1)),
    ],
)
def test_force_deterministic_rsag_routes_to_nccl(
    backend, monkeypatch, method_name, args
):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)

    getattr(backend, method_name)(*args)

    getattr(backend._nccl, method_name).assert_called_once_with(*args)
    getattr(backend._rsag, method_name).assert_not_called()


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("token_all_gather", (torch.empty(1, 4), (0, 1), [1, 1])),
        ("token_reduce_scatter", (torch.empty(2, 4), (0, 1), [1, 1])),
    ],
)
def test_default_token_ops_keep_triton_rsag(backend, monkeypatch, method_name, args):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)

    getattr(backend, method_name)(*args)

    getattr(backend._rsag, method_name).assert_called_once_with(*args)
    getattr(backend._nccl, method_name).assert_not_called()


def test_force_deterministic_rsag_routes_all_reduce_to_nccl(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    tensor = torch.empty(1, 4)
    group = (0, 1)

    backend.all_reduce(tensor, group)

    backend._nccl.all_reduce.assert_called_once_with(tensor, group, op=None)
    backend._trtllm_ar.has_trtllm_ar.assert_not_called()
    backend._triton_ar.can_run.assert_not_called()


def test_force_deterministic_rsag_routes_all_reduce_collection_to_nccl(
    backend, monkeypatch
):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    first = torch.empty(1, 4)
    second = torch.empty(1, 4)
    third = torch.empty(1, 8)
    group = (0, 1)

    tensors = (first, second, third)
    backend.all_reduce(tensors, group)

    assert backend._nccl.all_reduce.call_count == len(tensors)
    for call, tensor in zip(backend._nccl.all_reduce.call_args_list, tensors):
        assert call.args[0] is tensor
        assert call.args[1] == group
        assert call.kwargs == {"op": None}
    backend._nccl.all_reduce_two.assert_not_called()
    backend._trtllm_ar.has_trtllm_ar.assert_not_called()
    backend._triton_ar.can_run.assert_not_called()


def test_all_reduce_rejects_empty_collection(backend):
    with pytest.raises(ValueError, match="requires at least one tensor"):
        backend.all_reduce((), (0, 1))


def test_triton_collection_fallback_reduces_each_tensor(monkeypatch):
    fallback = Mock()
    fallback.all_reduce.side_effect = lambda tensor, _group, op: tensor
    backend = TritonAllReduceBackend(fallback)
    monkeypatch.setattr(backend, "_get_or_create", lambda _group: object())
    monkeypatch.setattr(
        triton_allreduce_module,
        "all_reduce_can_run",
        lambda _state, _tensor, op: False,
    )
    tensors = tuple(torch.empty(1) for _ in range(3))

    assert backend.all_reduce(tensors, (0, 1)) == tensors
    assert fallback.all_reduce.call_count == len(tensors)
    assert all(
        call.args[0] is tensor
        for call, tensor in zip(fallback.all_reduce.call_args_list, tensors)
    )


def test_triton_plan_does_not_initialize_iris_off_cdna4(monkeypatch):
    fallback = Mock()
    backend = TritonAllReduceBackend(fallback)
    get_or_create = Mock(side_effect=AssertionError("must not initialize Iris"))
    monkeypatch.setattr(backend, "_get_or_create", get_or_create)
    monkeypatch.setattr(
        triton_allreduce_module,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=False),
    )

    plan = backend.plan_all_reduce(((2, 4),), torch.empty(2, 4), (0, 1))

    assert tuple(output.shape for output in plan.outputs) == ((2, 4),)
    get_or_create.assert_not_called()


def test_force_deterministic_rsag_preserves_two_tensor_nccl_grouping(
    backend, monkeypatch
):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    tensors = (torch.empty(1, 4), torch.empty(1, 8))
    group = (0, 1)
    backend._nccl.all_reduce_two.return_value = tensors

    assert backend.all_reduce(tensors, group) is tensors
    backend._nccl.all_reduce_two.assert_called_once_with(*tensors, group, op=None)
    backend._nccl.all_reduce.assert_not_called()


def test_amd_collection_past_iris_capacity_uses_grouped_rccl(
    backend,
    monkeypatch,
):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    tensors = (
        torch.empty(384 * 1024, dtype=torch.bfloat16),
        torch.empty(256 * 1024, dtype=torch.bfloat16),
    )
    group = (0, 1)
    backend._nccl.all_reduce_two.return_value = tensors

    assert backend.all_reduce(tensors, group) is tensors
    backend._nccl.all_reduce_two.assert_called_once_with(*tensors, group, op=None)


def test_plan_all_reduce_uses_triton(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    backend._trtllm_ar.has_trtllm_ar.return_value = False
    backend._triton_ar.plan_all_reduce.return_value = "plan"
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    result = backend.plan_all_reduce(shapes, like, (0, 1))

    assert result == "plan"
    backend._triton_ar.plan_all_reduce.assert_called_once_with(
        shapes, like, (0, 1), op=None
    )
    backend._triton_ar.can_plan_all_reduce.assert_called_once_with(
        shapes, like, (0, 1), op=None
    )


def test_plan_all_reduce_amd_uses_base_when_iris_is_ineligible(
    backend,
    monkeypatch,
):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    backend._trtllm_ar.has_trtllm_ar.return_value = False
    backend._triton_ar.can_plan_all_reduce.return_value = False
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    result = backend.plan_all_reduce(shapes, like, (0, 1))

    assert tuple(output.shape for output in result.outputs) == shapes
    backend._triton_ar.plan_all_reduce.assert_not_called()


def test_plan_all_reduce_non_amd_keeps_existing_delegation(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=False),
    )
    backend._trtllm_ar.has_trtllm_ar.return_value = False
    backend._triton_ar.plan_all_reduce.return_value = "plan"
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    assert backend.plan_all_reduce(shapes, like, (0, 1)) == "plan"
    backend._triton_ar.plan_all_reduce.assert_called_once_with(
        shapes, like, (0, 1), op=None
    )
    backend._triton_ar.can_plan_all_reduce.assert_not_called()


def test_plan_all_reduce_preserves_trtllm(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    backend._trtllm_ar.has_trtllm_ar.return_value = True
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    result = backend.plan_all_reduce(shapes, like, (0, 1))

    assert tuple(output.shape for output in result.outputs) == shapes
    backend._triton_ar.plan_all_reduce.assert_not_called()
