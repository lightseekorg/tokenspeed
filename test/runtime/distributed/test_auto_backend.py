from unittest.mock import Mock, patch

import pytest
import torch

from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
from tokenspeed.runtime.distributed.comm_backend.base import CommBackend
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture
def backend(monkeypatch):
    instance = AutoBackend()
    monkeypatch.setattr(instance, "_nccl", Mock())
    monkeypatch.setattr(instance, "_rsag", Mock())
    monkeypatch.setattr(instance, "_custom_ar", Mock())
    monkeypatch.setattr(instance, "_trtllm_ar", Mock())
    monkeypatch.setattr(instance, "_triton_ar", Mock())
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
    backend._custom_ar.has_custom_ar.assert_not_called()
    backend._trtllm_ar.has_trtllm_ar.assert_not_called()
    backend._triton_ar.can_run.assert_not_called()


def test_force_deterministic_rsag_routes_all_reduce_two_to_nccl(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    first = torch.empty(1, 4)
    second = torch.empty(1, 4)
    group = (0, 1)

    backend.all_reduce_two(first, second, group)

    backend._nccl.all_reduce_two.assert_called_once_with(first, second, group, op=None)
    backend._custom_ar.has_custom_ar.assert_not_called()
    backend._trtllm_ar.has_trtllm_ar.assert_not_called()
    backend._triton_ar.can_run_two.assert_not_called()


def test_force_deterministic_rsag_disables_fused_attnres(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    tensor = torch.empty(1, 4)
    scratch = (torch.empty(1), torch.empty(1), torch.empty(1, 4))
    fallback_result = (Mock(), Mock())

    with patch.object(
        CommBackend,
        "all_reduce_residual_attnres",
        return_value=fallback_result,
    ) as fallback:
        result = backend.all_reduce_residual_attnres(
            tensor,
            tensor,
            torch.empty(4),
            torch.empty(4),
            scratch,
            1e-5,
            (0, 1),
        )

    assert result is fallback_result
    fallback.assert_called_once()
    backend._custom_ar.has_custom_ar.assert_not_called()
    backend._trtllm_ar.has_trtllm_ar.assert_not_called()
    backend._triton_ar.can_run_residual_attnres.assert_not_called()
