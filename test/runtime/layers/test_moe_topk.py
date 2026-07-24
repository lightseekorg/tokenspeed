from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe import topk as topk_module
from tokenspeed.runtime.layers.moe.topk import (
    TopKConfig,
    select_experts,
    torch_native_fused_topk,
)


@pytest.mark.parametrize("renormalize", [False, True])
def test_correction_bias_route_forwards_renormalize(
    monkeypatch: pytest.MonkeyPatch,
    renormalize: bool,
) -> None:
    calls: list[bool] = []

    def fake_cuda_routing_flash(
        _router_logits: torch.Tensor,
        _correction_bias: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        _num_real_experts: int,
        _routed_scaling_factor: float,
        renorm: bool,
    ) -> None:
        calls.append(renorm)
        topk_ids.fill_(0)
        topk_weights.fill_(1.0)

    monkeypatch.setattr(
        topk_module,
        "cuda_routing_flash",
        fake_cuda_routing_flash,
    )

    select_experts(
        hidden_states=torch.empty((1, 4), dtype=torch.float32),
        router_logits=torch.empty((1, 8), dtype=torch.float32),
        topk_config=TopKConfig(
            top_k=2,
            renormalize=renormalize,
            correction_bias=torch.zeros((8,), dtype=torch.float32),
            routed_scaling_factor=1.0,
        ),
    )

    assert calls == [renormalize]


def test_torch_native_fused_topk_breaks_exact_ties_by_expert_order() -> None:
    hidden_states = torch.zeros((2, 1), dtype=torch.float32)
    router_logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    topk_weights, topk_ids = torch_native_fused_topk(
        hidden_states,
        router_logits,
        topk=2,
        renormalize=True,
    )

    torch.testing.assert_close(topk_ids, torch.tensor([[0, 1], [0, 2]]))
    torch.testing.assert_close(
        topk_weights,
        torch.full((2, 2), 0.5, dtype=torch.float32),
    )


def test_torch_native_fused_topk_uses_biased_scores_only_for_choice() -> None:
    hidden_states = torch.zeros((1, 1), dtype=torch.float32)
    router_logits = torch.zeros((1, 4), dtype=torch.float32)
    correction_bias = torch.tensor([0.2, 0.1, 0.2, 0.0], dtype=torch.float32)

    topk_weights, topk_ids = torch_native_fused_topk(
        hidden_states,
        router_logits,
        topk=2,
        renormalize=True,
        correction_bias=correction_bias,
    )

    torch.testing.assert_close(topk_ids, torch.tensor([[0, 2]]))
    torch.testing.assert_close(
        topk_weights,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    )
