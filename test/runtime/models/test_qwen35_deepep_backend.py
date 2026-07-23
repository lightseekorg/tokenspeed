"""Qwen3.5 target and MTP draft DeepEP backend selection tests."""

from types import SimpleNamespace

import pytest
import torch

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
