from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe import topk as topk_module
from tokenspeed.runtime.layers.moe.topk import (
    TopK,
    TopKConfig,
    TopKOutputFormat,
    select_experts,
)


def test_topk_call_can_override_configured_output_format() -> None:
    topk = TopK(top_k=2, output_format=TopKOutputFormat.STANDARD)
    hidden_states = torch.empty((3, 4))
    router_logits = torch.empty((3, 8))

    output = topk(
        hidden_states,
        router_logits,
        output_format=TopKOutputFormat.BYPASSED,
    )

    assert output.format.is_bypassed()
    assert output.hidden_states is hidden_states
    assert output.router_logits is router_logits


def test_plain_route_uses_kernel_package_softmax_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool, float]] = []

    def fake_softmax_topk(
        router_logits: torch.Tensor,
        topk: int,
        *,
        renormalize: bool,
        routed_scaling_factor: float,
        enable_pdl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del enable_pdl
        calls.append((topk, renormalize, routed_scaling_factor))
        shape = (router_logits.shape[0], topk)
        return torch.ones(shape), torch.zeros(shape, dtype=torch.int64)

    monkeypatch.setattr(topk_module, "moe_softmax_topk", fake_softmax_topk)
    output = select_experts(
        hidden_states=torch.empty((2, 4), dtype=torch.float32),
        router_logits=torch.empty((2, 8), dtype=torch.float32),
        topk_config=TopKConfig(
            top_k=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        ),
    )

    assert calls == [(2, True, 2.5)]
    assert output.topk_weights.shape == output.topk_ids.shape == (2, 2)


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
