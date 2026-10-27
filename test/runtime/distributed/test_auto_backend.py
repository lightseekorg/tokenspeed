import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.distributed.comm_backend import iris as iris_module
from tokenspeed.runtime.distributed.comm_backend.auto import (
    AutoBackend,
    Collective,
    Route,
    ordered_fold_sum,
)
from tokenspeed.runtime.distributed.comm_backend.iris import (
    IrisBackend,
)
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
    TrtllmAllReduceBackend,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def test_runtime_import_without_iris():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import sys
                sys.modules["iris"] = None
                sys.modules["tokenspeed_kernel.ops.communication._iris"] = None
                from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
                from tokenspeed.runtime.layers.layernorm import RMSNorm
                AutoBackend()
            """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def backend(monkeypatch):
    instance = AutoBackend()
    monkeypatch.setattr(instance, "_nccl", Mock())
    monkeypatch.setattr(instance, "_rsag", Mock())
    monkeypatch.setattr(instance, "_trtllm_ar", Mock())
    monkeypatch.setattr(instance, "_iris", Mock())
    instance._iris.producer_direct_max_bytes = 1024 * 1024
    instance._iris.can_acquire_outputs.return_value = True
    instance._iris.can_reduce_outputs.return_value = False
    return instance


def test_all_reduce_rejects_empty_collection(backend):
    with pytest.raises(ValueError, match="requires at least one tensor"):
        backend.all_reduce((), (0, 1))


def test_iris_collection_fallback_reduces_each_tensor(monkeypatch):
    fallback = Mock()
    fallback.all_reduce.side_effect = lambda tensor, _group, op: tensor
    backend = IrisBackend(fallback)
    tensors = tuple(torch.empty(1) for _ in range(3))

    assert backend.all_reduce(tensors, (0, 1)) == tensors
    assert fallback.all_reduce.call_count == len(tensors)
    assert all(
        call.args[0] is tensor
        for call, tensor in zip(fallback.all_reduce.call_args_list, tensors)
    )


def test_iris_ordinary_all_reduce_keeps_512_kib_limit(monkeypatch):
    backend = IrisBackend(Mock(), producer_direct_max_bytes=1024 * 1024)
    group = tuple(range(8))
    tensor = Mock(
        is_cuda=True,
        is_contiguous=Mock(return_value=True),
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        iris_module,
        "current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )

    tensor.numel.return_value = 36 * 7168
    assert backend.can_run(tensor, group)
    tensor.numel.return_value = 37 * 7168
    assert not backend.can_run(tensor, group)
    assert backend.producer_direct_max_bytes == 1024 * 1024


def test_iris_preparation_caps_only_ordinary_staging(monkeypatch):
    backend = IrisBackend(Mock(), producer_direct_max_bytes=256)
    group = (0, 1)
    state = SimpleNamespace(
        staged_max_numel=128,
        producer_direct_max_numel=512,
        dtype=torch.bfloat16,
    )
    created = []

    def create(requested_group, dtype, staged, producer):
        created.append((requested_group, dtype, staged, producer))
        backend.install(requested_group, state)
        return state

    monkeypatch.setattr(
        iris_module, "current_platform", lambda: SimpleNamespace(is_amd=True)
    )
    monkeypatch.setattr(backend, "_create", create)
    capacities = dict(
        staged_max_numel=512, producer_direct_max_numel=512, dtype=torch.bfloat16
    )
    assert backend.prepare_all_reduce_buffers(group, **capacities)
    assert created == [(group, torch.bfloat16, 128, 512)]
    assert backend.prepare_all_reduce_buffers(group, **capacities)
    assert backend.state(group, torch.bfloat16) is state
    assert len(created) == 1


def test_public_preparation_preserves_capacities(backend, monkeypatch):
    from tokenspeed.runtime.distributed.comm_ops import prepare_all_reduce_buffers

    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    backend._trtllm_ar.has_trtllm_ar.return_value = False
    backend._iris.prepare_all_reduce_buffers.return_value = True
    capacities = dict(
        staged_max_numel=0, producer_direct_max_numel=512 * 10752, dtype=torch.bfloat16
    )
    group = tuple(range(8))
    assert prepare_all_reduce_buffers(group, **capacities, backend=backend)
    backend._iris.prepare_all_reduce_buffers.assert_called_once_with(
        group, **capacities
    )


def test_unprepared_group_keeps_default_producer_dispatch_limit(monkeypatch):
    backend = IrisBackend(Mock(), producer_direct_max_bytes=1024)
    monkeypatch.setattr(
        iris_module,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(
        backend,
        "_get_or_create",
        Mock(side_effect=AssertionError("capacity gate must run first")),
    )

    assert not backend.can_acquire_outputs(
        ((513,),),
        SimpleNamespace(is_cuda=True, dtype=torch.bfloat16),
        (0, 1),
    )


def test_prepared_group_uses_its_producer_capacity(monkeypatch):
    backend = IrisBackend(Mock(), producer_direct_max_bytes=1024)
    group = (0, 1)
    max_tokens = 8192
    shapes = ((max_tokens, 3584), (max_tokens, 7168))
    state = SimpleNamespace(
        producer_direct_max_numel=max_tokens * (3584 + 7168), dtype=torch.bfloat16
    )
    backend._instances[group, torch.bfloat16] = state
    monkeypatch.setattr(
        iris_module,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    like = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)

    assert backend.can_acquire_outputs(shapes, like, group)
    assert not backend.can_acquire_outputs(
        ((max_tokens + 1, 3584), (max_tokens + 1, 7168)),
        like,
        group,
    )


def test_iris_output_acquisition_does_not_initialize_iris_off_cdna4(monkeypatch):
    fallback = Mock()
    backend = IrisBackend(fallback)
    get_or_create = Mock(side_effect=AssertionError("must not initialize Iris"))
    monkeypatch.setattr(backend, "_get_or_create", get_or_create)
    monkeypatch.setattr(
        iris_module,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=False),
    )

    outputs = backend.acquire_all_reduce_outputs(((2, 4),), torch.empty(2, 4), (0, 1))

    assert tuple(output.shape for output in outputs) == ((2, 4),)
    get_or_create.assert_not_called()


def test_iris_output_acquisition_propagates_iris_setup_failure(monkeypatch):
    fallback = Mock()
    backend = IrisBackend(fallback)
    monkeypatch.setattr(
        iris_module,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    monkeypatch.setattr(
        backend,
        "_get_or_create",
        Mock(side_effect=RuntimeError("Iris setup failed")),
    )
    like = Mock(is_cuda=True, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="Iris setup failed"):
        backend.acquire_all_reduce_outputs(((2, 4),), like, (0, 1))


def test_nccl_collection_uses_single_tensor_path():
    backend = NcclBackend()
    backend._resources[(0,)] = {
        "pynccl_comm": None,
        "device_group": None,
        "world_size": 1,
    }
    tensors = (torch.empty(1), torch.empty(2), torch.empty(3))

    outputs = backend.all_reduce(tensors, (0,))

    assert all(output is tensor for output, tensor in zip(outputs, tensors))


def test_trtllm_collection_uses_fallback_single_tensor_path():
    fallback = Mock()
    fallback.all_reduce.side_effect = lambda tensor, _group, op: tensor
    backend = TrtllmAllReduceBackend(fallback)
    tensors = (torch.empty(1), torch.empty(2), torch.empty(3))

    outputs = backend.all_reduce(tensors, (0, 1))

    assert all(output is tensor for output, tensor in zip(outputs, tensors))
    assert fallback.all_reduce.call_count == len(tensors)


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


def test_acquire_all_reduce_outputs_uses_iris(backend, monkeypatch):
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    backend._trtllm_ar.has_trtllm_ar.return_value = False
    expected = (torch.empty(1, 7168), torch.empty(1, 3584))
    backend._iris.acquire_all_reduce_outputs.return_value = expected
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    result = backend.acquire_all_reduce_outputs(shapes, like, (0, 1))

    assert result is expected
    backend._iris.acquire_all_reduce_outputs.assert_called_once_with(
        shapes, like, (0, 1), op=None
    )
    backend._iris.can_acquire_outputs.assert_called_once_with(
        shapes, like, (0, 1), op=None
    )


def test_acquire_all_reduce_outputs_amd_uses_base_when_iris_is_ineligible(
    backend,
    monkeypatch,
):
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    backend._trtllm_ar.has_trtllm_ar.return_value = False
    backend._iris.can_acquire_outputs.return_value = False
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    result = backend.acquire_all_reduce_outputs(shapes, like, (0, 1))

    assert tuple(output.shape for output in result) == shapes
    backend._iris.acquire_all_reduce_outputs.assert_not_called()


def test_acquire_all_reduce_outputs_non_amd_keeps_existing_delegation(
    backend, monkeypatch
):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=False),
    )
    backend._trtllm_ar.has_trtllm_ar.return_value = False
    expected = (torch.empty(1, 7168), torch.empty(1, 3584))
    backend._iris.acquire_all_reduce_outputs.return_value = expected
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    assert backend.acquire_all_reduce_outputs(shapes, like, (0, 1)) is expected
    backend._iris.acquire_all_reduce_outputs.assert_called_once_with(
        shapes, like, (0, 1), op=None
    )
    backend._iris.can_acquire_outputs.assert_not_called()


def test_acquire_all_reduce_outputs_preserves_trtllm(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    backend._trtllm_ar.has_trtllm_ar.return_value = True
    like = torch.empty(1, 3584, dtype=torch.bfloat16)
    shapes = ((1, 7168), (1, 3584))

    result = backend.acquire_all_reduce_outputs(shapes, like, (0, 1))

    assert tuple(output.shape for output in result) == shapes
    backend._iris.acquire_all_reduce_outputs.assert_not_called()


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_symmetric_outputs_route_back_to_iris(backend, monkeypatch, num_outputs):
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    backend._iris.can_reduce_outputs.return_value = True
    outputs = (torch.empty(1, 4), torch.empty(1, 8))[:num_outputs]
    backend._iris.all_reduce.return_value = outputs

    assert backend.all_reduce(outputs, (0, 1)) is outputs
    backend._iris.all_reduce.assert_called_once_with(outputs, (0, 1), op=None)
    backend._nccl.all_reduce.assert_not_called()


def test_prepared_symmetric_outputs_bypass_default_size_gate(backend, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", False)
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.comm_backend.auto.current_platform",
        lambda: SimpleNamespace(is_amd=True),
    )
    backend._iris.can_reduce_outputs.return_value = True
    outputs = (
        torch.empty(2 * 1024 * 1024, dtype=torch.bfloat16),
        torch.empty(2 * 1024 * 1024, dtype=torch.bfloat16),
    )
    backend._iris.all_reduce.return_value = outputs

    assert backend.all_reduce(outputs, (0, 1)) is outputs
    backend._iris.all_reduce.assert_called_once_with(outputs, (0, 1), op=None)
    backend._nccl.all_reduce_two.assert_not_called()


# ---- The one routing decision ----------------------------------------------


@pytest.fixture
def routing(backend, monkeypatch):
    """The fixture backend with a controllable multicast verdict; ``mapping``
    None makes every group node-local."""
    monkeypatch.setitem(global_server_args_dict, "mapping", None)
    return backend


def _switches(monkeypatch, *, force: bool, invariant: bool) -> None:
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", force)
    monkeypatch.setitem(
        global_server_args_dict, "batch_invariant_collectives", invariant
    )


ROWS = torch.empty(4, 16, dtype=torch.bfloat16)
GROUP = (0, 1)
REDUCTIONS = [collective for collective in Collective if collective.is_reduction]


@pytest.mark.parametrize("collective", list(Collective))
def test_force_deterministic_rsag_routes_everything_to_nccl(
    routing, monkeypatch, collective
):
    _switches(monkeypatch, force=True, invariant=False)
    assert routing.route(collective, ROWS, GROUP) is Route.NCCL


@pytest.mark.parametrize("collective", list(Collective))
def test_force_deterministic_rsag_keeps_reductions_on_the_fold(
    routing, monkeypatch, collective
):
    _switches(monkeypatch, force=True, invariant=True)
    expected = Route.ORDERED_FOLD if collective.is_reduction else Route.NCCL
    assert routing.route(collective, ROWS, GROUP) is expected


@pytest.mark.parametrize("collective", REDUCTIONS)
@pytest.mark.parametrize("reachable", [True, False])
@pytest.mark.parametrize(
    "payload",
    [
        torch.empty(4, 16, dtype=torch.bfloat16),
        torch.empty(4, 16, dtype=torch.float32),
        torch.empty(64, dtype=torch.bfloat16),
        torch.empty(4, 12, dtype=torch.bfloat16),
    ],
)
def test_batch_invariant_reductions_always_fold(
    routing, monkeypatch, collective, reachable, payload
):
    # The in-switch reduction's order moves with a row's offset in the buffer,
    # so no reduction takes it under the contract, whatever multicast reaches.
    _switches(monkeypatch, force=False, invariant=True)
    monkeypatch.setattr(
        AutoBackend, "_multicast_reachable", staticmethod(lambda g: reachable)
    )
    assert routing.route(collective, payload, GROUP) is Route.ORDERED_FOLD


@pytest.mark.parametrize(
    "collective", [Collective.ALL_GATHER, Collective.TOKEN_ALL_GATHER]
)
@pytest.mark.parametrize("invariant", [False, True])
def test_gathers_follow_multicast_reachability_whatever_the_envelope(
    routing, monkeypatch, collective, invariant
):
    _switches(monkeypatch, force=False, invariant=invariant)
    assert routing.route(collective, ROWS, GROUP) is Route.MULTIMEM
    monkeypatch.setattr(
        AutoBackend, "_multicast_reachable", staticmethod(lambda g: False)
    )
    assert routing.route(collective, ROWS, GROUP) is Route.NCCL


def test_auto_routes_keep_the_performance_defaults(routing, monkeypatch):
    _switches(monkeypatch, force=False, invariant=False)
    assert routing.route(Collective.ALL_REDUCE, ROWS, GROUP) is Route.TIERED
    assert routing.route(Collective.REDUCE_SCATTER, ROWS, GROUP) is Route.NCCL
    assert routing.route(Collective.TOKEN_REDUCE_SCATTER, ROWS, GROUP) is Route.MULTIMEM


def test_all_gather_takes_the_workspace_exchange_where_it_serves(routing, monkeypatch):
    _switches(monkeypatch, force=False, invariant=False)
    gathered = torch.empty(4, 32, dtype=torch.bfloat16)
    routing._trtllm_ar.lamport_all_gather.return_value = gathered
    assert routing.all_gather(ROWS, GROUP, -1) is gathered
    routing._rsag.all_gather.assert_not_called()
    routing._trtllm_ar.lamport_all_gather.return_value = None
    assert routing.all_gather(ROWS, GROUP, -1) is routing._rsag.all_gather.return_value
    routing._rsag.all_gather.assert_called_once_with(ROWS, GROUP, -1)


def test_route_never_reads_the_row_count(routing, monkeypatch):
    # A site's route must not move with the batch.
    _switches(monkeypatch, force=False, invariant=True)
    for rows in (0, 1, 7, 100_000):
        payload = torch.empty(rows, 16, dtype=torch.bfloat16)
        assert (
            routing.route(Collective.ALL_REDUCE, payload, GROUP) is Route.ORDERED_FOLD
        )


def test_batch_invariant_all_reduce_folds_the_gathered_partials_in_rank_order(
    routing, monkeypatch
):
    _switches(monkeypatch, force=False, invariant=True)
    routing._trtllm_ar.has_trtllm_ar.return_value = False
    monkeypatch.setattr(
        AutoBackend, "_multicast_reachable", staticmethod(lambda g: False)
    )
    # Two partials whose sum rounds differently in bf16 than in fp32: the fold
    # accumulates in fp32 and rounds once.
    mine = torch.full((4, 16), 1.0, dtype=torch.bfloat16)
    peer = torch.full((4, 16), 2.0**-9, dtype=torch.bfloat16)
    routing._nccl.all_gather.side_effect = lambda tensor, group, dim: torch.cat(
        [mine, peer], dim=dim
    )
    tensor = mine.clone()

    assert routing.all_reduce(tensor, GROUP) is tensor

    expected = (mine.float() + peer.float()).to(torch.bfloat16)
    assert torch.equal(tensor, expected)
    routing._nccl.all_reduce.assert_not_called()
    assert not routing._rsag.method_calls
    # Collections take the same route tensor by tensor.
    routing._nccl.all_gather.reset_mock()
    routing.all_reduce((mine.clone(), mine.clone()), GROUP)
    assert routing._nccl.all_gather.call_count == 2


def test_batch_invariant_all_reduce_takes_the_trtllm_fold_where_it_serves(
    routing, monkeypatch
):
    _switches(monkeypatch, force=False, invariant=True)
    monkeypatch.setattr(
        AutoBackend, "_multicast_reachable", staticmethod(lambda g: False)
    )
    routing._trtllm_ar.has_trtllm_ar.return_value = True
    folded = torch.full((4, 16), 3.0, dtype=torch.bfloat16)
    routing._trtllm_ar.ordered_all_reduce.return_value = folded
    routing._nccl.all_gather.side_effect = lambda tensor, group, dim: torch.cat(
        [tensor, tensor], dim=dim
    )
    gather_fold = torch.full((4, 16), 2.0, dtype=torch.bfloat16)
    assert routing.all_reduce(torch.ones(4, 16, dtype=torch.bfloat16), GROUP) is folded

    # The rsag override keeps the fold off the symmetric-memory workspace.
    _switches(monkeypatch, force=True, invariant=True)
    tensor = torch.ones(4, 16, dtype=torch.bfloat16)
    assert routing.all_reduce(tensor, GROUP) is tensor
    assert torch.equal(tensor, gather_fold)

    # A shape the workspace does not serve takes the gather and the fold.
    _switches(monkeypatch, force=False, invariant=True)
    routing._trtllm_ar.ordered_all_reduce.return_value = None
    tensor = torch.ones(4, 16, dtype=torch.bfloat16)
    assert routing.all_reduce(tensor, GROUP) is tensor
    assert torch.equal(tensor, gather_fold)


def test_ordered_fold_sum_stores_a_zero_sum_as_positive_zero():
    # The torch fold matches the kernels: only an all -0.0 sum would keep its sign.
    parts = torch.tensor([[-0.0, -0.0, 1.0], [-0.0, 0.0, -1.0]])
    out = torch.empty(3)
    ordered_fold_sum(parts, out)
    assert out.view(torch.int32).tolist() == [0, 0, 0]


def test_batch_invariant_all_reduce_refuses_order_dependent_non_sum(
    routing, monkeypatch
):
    # MAX and MIN are exact in any order and take NCCL; a floating-point
    # product depends on the order and has no batch-invariant route.
    _switches(monkeypatch, force=False, invariant=True)
    with pytest.raises(ValueError, match="SUM"):
        routing.all_reduce(
            torch.ones(4, 16, dtype=torch.bfloat16),
            GROUP,
            op=torch.distributed.ReduceOp.PRODUCT,
        )
