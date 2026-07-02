"""CPU-side tests for TRT-LLM all-reduce selection and runtime wiring."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
    TrtllmAllReduceBackend,
)
from tokenspeed.runtime.execution.distributed_initializer import (
    DistributedInitializer,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def _fake_cuda_bf16_tensor(rows: int, columns: int):
    tensor = MagicMock()
    tensor.is_cuda = True
    tensor.dtype = torch.bfloat16
    tensor.shape = (rows, columns)
    tensor.is_contiguous.return_value = True
    tensor.dim.return_value = 2
    tensor.numel.return_value = rows * columns
    tensor.element_size.return_value = 2
    return tensor


def test_server_flag_is_explicit_opt_in():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    default_args = parser.parse_args(["--model=stub"])
    enabled_args = parser.parse_args(["--model=stub", "--enable-trtllm-allreduce"])

    assert default_args.enable_trtllm_allreduce is False
    assert enabled_args.enable_trtllm_allreduce is True


def test_auto_backend_selects_configured_trtllm_backend():
    backend = AutoBackend()
    group = (0, 1)
    tensor = object()
    expected = object()

    backend._custom_ar = SimpleNamespace(has_custom_ar=lambda _group: False)
    backend._trtllm_ar = SimpleNamespace(
        has_trtllm_ar=lambda _group: True,
        all_reduce=lambda _tensor, _group, op=None: expected,
    )
    backend._triton_ar = SimpleNamespace(can_run=lambda *_args, **_kwargs: False)
    backend._nccl = SimpleNamespace(
        all_reduce=lambda *_args, **_kwargs: pytest.fail("unexpected NCCL fallback")
    )

    assert backend.all_reduce(tensor, group) is expected


@pytest.mark.parametrize(
    ("use_fp32_lamport", "expected_max_tokens"),
    ((False, 146), (True, 73)),
)
def test_configure_group_clamps_workspace_to_oneshot_limit(
    monkeypatch, use_fp32_lamport, expected_max_tokens
):
    from tokenspeed.runtime.distributed import process_group_manager
    from tokenspeed.runtime.distributed.comm_backend import trtllm_allreduce

    backend = TrtllmAllReduceBackend(fallback=MagicMock())
    group = tuple(range(8))
    calls = []

    monkeypatch.setattr(backend, "_load_comm", lambda: True)
    monkeypatch.setattr(
        process_group_manager.process_group_manager,
        "get_process_group",
        lambda kind, requested_group: (kind, requested_group),
    )

    def create_workspace(rank, size, max_tokens, hidden_dim, **kwargs):
        calls.append((rank, size, max_tokens, hidden_dim, kwargs))
        return [[1]], object()

    monkeypatch.setattr(
        trtllm_allreduce,
        "trtllm_create_ipc_workspace_for_all_reduce_fusion",
        create_workspace,
    )

    assert backend.configure_group(
        rank=3,
        group=group,
        max_token_num=8192,
        hidden_dim=7168,
        use_fp32_lamport=use_fp32_lamport,
    )
    assert calls[0][2] == expected_max_tokens
    assert calls[0][4]["use_fp32_lamport"] is use_fp32_lamport
    assert backend._resources[group]["max_token_num"] == expected_max_tokens


def test_backend_eligibility_is_fail_closed():
    backend = TrtllmAllReduceBackend(fallback=MagicMock())
    group = tuple(range(8))
    backend._resources[group] = {
        "max_token_num": 146,
        "hidden_dim": 7168,
    }

    assert backend.can_run(_fake_cuda_bf16_tensor(16, 7168), group)
    assert not backend.can_run(_fake_cuda_bf16_tensor(147, 7168), group)

    fp32 = _fake_cuda_bf16_tensor(16, 7168)
    fp32.dtype = torch.float32
    assert not backend.can_run(fp32, group)

    noncontiguous = _fake_cuda_bf16_tensor(16, 7168)
    noncontiguous.is_contiguous.return_value = False
    assert not backend.can_run(noncontiguous, group)

    one_dimensional = _fake_cuda_bf16_tensor(16, 7168)
    one_dimensional.dim.return_value = 1
    assert not backend.can_run(one_dimensional, group)

    three_dimensional = _fake_cuda_bf16_tensor(16, 7168)
    three_dimensional.dim.return_value = 3
    assert not backend.can_run(three_dimensional, group)


def test_distributed_wiring_deduplicates_alias_groups(monkeypatch):
    from tokenspeed.runtime.distributed.comm_backend import registry

    backend = AutoBackend()
    configure_group = MagicMock(return_value=True)
    backend._trtllm_ar = SimpleNamespace(configure_group=configure_group)
    monkeypatch.setattr(registry, "get_global_backend", lambda: backend)

    group = tuple(range(8))
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_group=group),
        dense=SimpleNamespace(tp_group=group),
        moe=SimpleNamespace(ep_size=1, tp_ep_group=group),
    )
    config = SimpleNamespace(
        enable_trtllm_allreduce=True,
        nnodes=1,
        mapping=mapping,
        global_rank=5,
        max_num_tokens=8192,
        hidden_size=7168,
    )

    DistributedInitializer._configure_trtllm_allreduce(config)

    configure_group.assert_called_once_with(
        rank=5,
        group=group,
        max_token_num=8192,
        hidden_dim=7168,
    )


def test_distributed_wiring_skips_expert_parallel_group(monkeypatch):
    from tokenspeed.runtime.distributed.comm_backend import registry

    backend = AutoBackend()
    configure_group = MagicMock(return_value=True)
    backend._trtllm_ar = SimpleNamespace(configure_group=configure_group)
    monkeypatch.setattr(registry, "get_global_backend", lambda: backend)

    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_group=(0,)),
        dense=SimpleNamespace(tp_group=(0,)),
        moe=SimpleNamespace(ep_size=8, tp_ep_group=tuple(range(8))),
    )
    config = SimpleNamespace(
        enable_trtllm_allreduce=True,
        nnodes=1,
        mapping=mapping,
        global_rank=0,
        max_num_tokens=8192,
        hidden_size=7168,
    )

    DistributedInitializer._configure_trtllm_allreduce(config)

    configure_group.assert_not_called()


def test_distributed_wiring_rejects_multinode():
    config = SimpleNamespace(enable_trtllm_allreduce=True, nnodes=2)
    with pytest.raises(RuntimeError, match="single-node"):
        DistributedInitializer._configure_trtllm_allreduce(config)
