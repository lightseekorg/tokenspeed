"""Qwen3.5 target and MTP draft DeepEP backend selection tests."""

from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.distributed.comm_manager as comm_manager_module
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.models import qwen3_5_moe


def _set_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    all2all: All2AllBackend,
    moe: MoeBackend,
) -> None:
    monkeypatch.setattr(qwen3_5_moe, "get_all2all_backend", lambda: all2all)
    monkeypatch.setattr(qwen3_5_moe, "get_moe_backend", lambda: moe)


def test_qwen35_deepep_target_uses_deepep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_backends(
        monkeypatch,
        all2all=All2AllBackend.DEEPEP,
        moe=MoeBackend.FLASHINFER_CUTEDSL,
    )

    assert qwen3_5_moe._qwen35_moe_a2a_backend() == "deepep"


def test_qwen35_mtp_draft_does_not_inherit_deepep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_backends(
        monkeypatch,
        all2all=All2AllBackend.DEEPEP,
        moe=MoeBackend.FLASHINFER_TRTLLM,
    )

    assert qwen3_5_moe._qwen35_moe_a2a_backend() == "none"


def test_qwen35_deepep_shared_expert_gathers_then_scatters_idle_rank() -> None:
    events = []
    local_hidden = torch.empty((0, 4))
    gathered_hidden = torch.ones((1, 4))

    class CommManager:
        def pre_dense_comm(self, hidden_states, ctx):
            assert hidden_states.shape == local_hidden.shape
            events.append("gather")
            return gathered_hidden

        def post_dense_comm(self, hidden_states, residual, ctx):
            assert hidden_states.shape == (1, 4)
            assert residual is None
            events.append("scatter")
            return local_hidden, None

    class TopK:
        @staticmethod
        def empty_topk_output(device, *, hidden_states, router_logits):
            return SimpleNamespace(
                topk_ids=torch.empty((0, 1), dtype=torch.int64),
                topk_weights=torch.empty((0, 1)),
                router_logits=router_logits,
            )

    def shared_expert(hidden_states):
        assert hidden_states is gathered_hidden
        events.append("shared_expert")
        return hidden_states

    def experts(**kwargs):
        assert kwargs["hidden_states"].shape == local_hidden.shape
        events.append("routed_experts")
        return local_hidden

    block = SimpleNamespace(
        gate=lambda hidden_states: (torch.empty((0, 8)), None),
        shared_expert=shared_expert,
        shared_expert_gate=None,
        comm_manager=CommManager(),
        topk=TopK(),
        experts=experts,
    )

    output = qwen3_5_moe.Qwen3_5MoeSparseMoeBlock._forward_deepep(
        block,
        local_hidden,
        num_global_tokens=1,
        max_num_tokens_per_gpu=1,
        ctx=object(),
    )

    assert output.shape == (0, 4)
    assert events == ["gather", "shared_expert", "scatter", "routed_experts"]


def test_qwen35_deepep_shared_expert_uses_enclosing_moe_row_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal attention/dense TP still needs RSAG when MoE rows are scattered."""
    mapping = Mapping(
        rank=0,
        world_size=4,
        attn_tp_size=2,
        attn_cp_size=1,
        attn_dp_size=2,
        dense_tp_size=2,
        dense_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=4,
        moe_dp_size=1,
    )
    manager = CommManager(
        mapping=mapping,
        layer_id=1,
        is_moe=True,
        prev_is_moe=True,
    )
    ctx = SimpleNamespace(
        collective_global_num_tokens=None,
        global_num_tokens=[3, 3, 5, 5],
        collective_num_tokens=None,
        input_num_tokens=3,
    )
    local_hidden = torch.ones((2, 4))
    gathered_hidden = torch.ones((3, 4))
    calls = []

    def fake_all_gather(hidden_states, *, group, scattered_num_tokens):
        assert hidden_states is local_hidden
        assert group == (0, 1)
        assert scattered_num_tokens == [2, 1]
        calls.append("gather")
        return gathered_hidden

    def fake_reduce_scatter(hidden_states, *, group, scattered_num_tokens):
        assert hidden_states is gathered_hidden
        assert group == (0, 1)
        assert scattered_num_tokens == [2, 1]
        calls.append("scatter")
        return local_hidden

    def unexpected_all_reduce(*args, **kwargs):
        pytest.fail("scattered MoE rows must not use dense-TP all-reduce")

    monkeypatch.setattr(comm_manager_module, "token_all_gather", fake_all_gather)
    monkeypatch.setattr(
        comm_manager_module, "token_reduce_scatter", fake_reduce_scatter
    )
    monkeypatch.setattr(comm_manager_module, "all_reduce", unexpected_all_reduce)

    shared_input = manager.pre_dense_comm(local_hidden, ctx)
    shared_output, residual = manager.post_dense_comm(shared_input, None, ctx)

    assert shared_output is local_hidden
    assert residual is None
    assert calls == ["gather", "scatter"]
