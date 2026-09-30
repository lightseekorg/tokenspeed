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

"""K3 routed-workspace capacity, input contracts, and second-stage assembly."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.models import kimi_k3_comm as mod


@pytest.fixture(autouse=True)
def reset_workspace(monkeypatch):
    monkeypatch.setattr(mod.K3MoeTailComm, "_routed_workspace", None)
    monkeypatch.setattr(mod.K3MoeTailComm, "_workspace_config", None)
    monkeypatch.setattr(mod.K3MoeTailComm, "_latent_tail", None)
    monkeypatch.setattr(mod.K3MoeTailComm, "_stage2_capacity", None)
    monkeypatch.setattr(mod.K3MoeTailComm, "_multimem_group_name", None)


def _comm(*, enabled, capacity, deferred):
    comm = object.__new__(mod.K3MoeTailComm)
    workspace = SimpleNamespace(
        max_num_tokens=capacity,
        supports_num_tokens=lambda m: 0 < m <= capacity,
    )
    mod.K3MoeTailComm._routed_workspace = workspace
    mod.K3MoeTailComm._stage2_capacity = capacity
    comm.use_allreduce_fusion = enabled
    comm.defer_finalize = enabled and deferred
    comm.routed_norm = SimpleNamespace(weight=object())
    return comm


def test_prepared_capacity_covers_tokens_above_8192(monkeypatch):
    comm = _comm(enabled=True, capacity=16384, deferred=True)
    output = object()
    reduce = Mock(return_value=output)
    monkeypatch.setattr(mod, "allreduce_fusion", reduce)
    routed = (object(), object(), object())
    assert comm.routed_ar_fusion(routed, 8193) is output
    assert reduce.call_args.kwargs["num_tokens"] == 8193
    with pytest.raises(ValueError, match="exceeds"):
        comm.routed_ar_fusion(routed, 16385)
    reduce.assert_called_once()


@pytest.mark.parametrize(
    "tp,ep,capacity",
    [
        (4, 1, 32768),
        (8, 1, 32768),
        (16, 1, 32768),
        (1, 4, 32768),
        (1, 8, 32768),
        (1, 16, 32768),
        (8, 1, 1024),
        (8, 1, 1025),
    ],
)
def test_workspace_uses_serving_capacity_once(monkeypatch, tp, ep, capacity):
    group_size = tp * ep
    group = SimpleNamespace(group_name=f"moe_tp{tp}_ep{ep}")
    workspace = SimpleNamespace(max_num_tokens=capacity)
    create = Mock(return_value=workspace)
    small = Mock()
    prealloc = Mock(return_value=True)
    monkeypatch.setattr(mod, "KimiK3LatentTailOp", small)
    monkeypatch.setattr(mod, "multimem_prealloc", prealloc)
    monkeypatch.setattr(mod, "_get_process_group", lambda ranks: group)
    monkeypatch.setattr(mod, "create_allreduce_fusion_workspace", create)
    monkeypatch.setattr(
        mod, "current_platform", lambda: SimpleNamespace(is_blackwell=True)
    )
    monkeypatch.setattr(mod, "global_server_args_dict", {"disable_pdl": True})
    mapping = SimpleNamespace(
        moe=SimpleNamespace(
            tp_size=tp,
            ep_size=ep,
            tp_ep_size=group_size,
            tp_ep_group=tuple(range(group_size)),
        ),
        attn=SimpleNamespace(tp_size=group_size),
    )
    comms = [
        mod.K3MoeTailComm(
            mapping=mapping,
            hidden_size=7168,
            routed_hidden=3584,
            top_k=16,
            routed_norm=SimpleNamespace(variance_epsilon=1e-5),
            up_proj=SimpleNamespace(shard_group=mapping.moe.tp_ep_group),
            experts_supports_deferred_finalize=True,
        )
        for _ in range(2)
    ]
    assert comms[0].prepare(capacity)
    assert comms[1]._routed_workspace is workspace
    assert comms[1].prepare(capacity)
    assert comms[1].prepare(min(8192, capacity))
    create.assert_called_once_with(
        group=group, hidden_size=3584, top_k=16, max_num_tokens=capacity, rms_eps=1e-5
    )
    small.assert_called_once_with(group=group, hidden_size=7168, latent_size=3584)
    if capacity > 1024:
        prealloc.assert_called_once_with(capacity, (7168,), group.group_name)
    else:
        prealloc.assert_not_called()
        assert comms[0]._multimem_group_name is None
    with pytest.raises(RuntimeError, match="grow"):
        comms[1].prepare(capacity + 1)


def test_model_prepares_routed_workspace_at_serving_limit(monkeypatch):
    from tokenspeed.runtime.models import kimi_k3

    layers = []
    for _ in range(2):
        moe = object.__new__(kimi_k3.KimiLinearMoE)
        torch.nn.Module.__init__(moe)
        moe.shared_experts = SimpleNamespace(shared_parallel=None)
        moe.comm = SimpleNamespace(prepare=Mock(return_value=True))
        layer = object.__new__(kimi_k3.KimiLinearDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.is_moe_layer = True
        layer.block_sparse_moe = moe
        layers.append(layer)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(layers)
    owner = SimpleNamespace(
        model=model,
        mapping=object(),
        config=SimpleNamespace(hidden_size=7168, routed_expert_hidden_size=3584),
    )
    monkeypatch.setattr(kimi_k3, "prepare_k3_all_reduce_buffers", lambda **kw: False)
    assert kimi_k3.KimiLinearForCausalLM.prepare_communication_runtime(owner, 32768)
    layers[0].block_sparse_moe.comm.prepare.assert_called_once_with(32768)
    layers[1].block_sparse_moe.comm.prepare.assert_not_called()


@pytest.mark.parametrize(
    "enabled,deferred", [(True, False), (True, True), (False, False)]
)
def test_routed_implementations_use_same_second_stage(monkeypatch, enabled, deferred):
    m = 33
    local_routed = torch.full((m, 3), 2, dtype=torch.bfloat16)
    summed_routed = torch.full_like(local_routed, 4)
    normalized = torch.ones_like(local_routed)
    shared = torch.ones(m, 16, dtype=torch.bfloat16)
    residual = torch.full_like(shared, 3)
    reduced = torch.full_like(shared, 21)
    fused_reduce = Mock(return_value=normalized)
    monkeypatch.setattr(mod, "allreduce_fusion", fused_reduce)
    group = tuple(range(8))

    def reduce(value, actual_group):
        assert actual_group == group
        if value.shape[-1] == 3:
            assert value is local_routed
            return summed_routed
        assert torch.all(value[:, :4] == 1)
        assert torch.all(value[:, 4:6] == 10)
        assert torch.all(value[:, 6:] == 1)
        return reduced

    plain_reduce = Mock(side_effect=reduce)
    monkeypatch.setattr(mod, "all_reduce", plain_reduce)
    comm = _comm(enabled=enabled, capacity=16384, deferred=deferred)
    comm.hidden_size = 16
    comm.routed_norm = Mock(
        return_value=normalized, weight=torch.ones(3, dtype=torch.bfloat16)
    )
    comm.up_proj = SimpleNamespace(
        shard_slice=(4, 2), weight=torch.full((2, 3), 2, dtype=torch.bfloat16)
    )
    comm.mapping = SimpleNamespace(
        moe=SimpleNamespace(has_tp_ep=True, tp_ep_size=8, tp_ep_group=group)
    )
    values = (local_routed, object(), object())
    routed_latent = comm.routed_ar_fusion(values if deferred else local_routed, m)
    result = comm.up_proj_inject_ar(routed_latent, shared, residual)
    if enabled:
        fused_reduce.assert_called_once_with(
            local_routed,
            comm._routed_workspace,
            pattern=(
                mod.AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
                if deferred
                else mod.AllReduceFusionPattern.ALLREDUCE_RMSNORM
            ),
            rms_gamma=comm.routed_norm.weight,
            num_tokens=m,
            expert_weights=values[1] if deferred else None,
            expanded_idx_to_permuted_idx=values[2] if deferred else None,
        )
        comm.routed_norm.assert_not_called()
        plain_reduce.assert_called_once()
    else:
        fused_reduce.assert_not_called()
        comm.routed_norm.assert_called_once_with(summed_routed)
        assert plain_reduce.call_count == 2
    assert result.data_ptr() == reduced.data_ptr()
    torch.testing.assert_close(result, reduced, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,use_multimem",
    [(256, False), (512, False), (1024, False), (1025, True), (8193, True)],
)
def test_up_proj_inject_ar_selects_collective(monkeypatch, m, use_multimem):
    comm = _comm(enabled=False, capacity=16384, deferred=False)
    comm.hidden_size = 16
    comm.mapping = SimpleNamespace(moe=SimpleNamespace(tp_ep_group=tuple(range(8))))
    comm.up_proj = SimpleNamespace(
        shard_slice=(4, 2), weight=torch.full((2, 3), 2, dtype=torch.bfloat16)
    )
    latent = torch.ones(m, 3, dtype=torch.bfloat16)
    shared = torch.ones(m, 16, dtype=torch.bfloat16)
    residual = torch.full_like(shared, 3)
    result = torch.full_like(shared, 21)
    monkeypatch.setattr(mod.K3MoeTailComm, "_multimem_group_name", "moe_tp8")
    stage = Mock(side_effect=lambda value, group, capacity: value.clone())

    def reduce(value, group):
        assert group == ("moe_tp8" if use_multimem else tuple(range(8)))
        assert torch.all(value[:, :4] == 1)
        assert torch.all(value[:, 4:6] == 10)
        assert torch.all(value[:, 6:] == 1)
        return result

    ordinary = Mock(side_effect=reduce)
    multimem = Mock(side_effect=reduce)
    monkeypatch.setattr(mod, "all_reduce", ordinary)
    monkeypatch.setattr(mod, "multimem_stage", stage)
    monkeypatch.setattr(mod, "multimem_all_reduce_staged", multimem)
    output = comm.up_proj_inject_ar(latent, shared, residual)
    if use_multimem:
        ordinary.assert_not_called()
        stage.assert_called_once()
        torch.testing.assert_close(stage.call_args.args[0], shared, rtol=0, atol=0)
        assert stage.call_args.args[1:] == ("moe_tp8", 16384)
        multimem.assert_called_once()
        result.zero_()
    else:
        ordinary.assert_called_once()
        multimem.assert_not_called()
        stage.assert_not_called()
    torch.testing.assert_close(output, torch.full_like(output, 21), rtol=0, atol=0)
