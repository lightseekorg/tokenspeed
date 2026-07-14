from __future__ import annotations

import pytest
import tokenspeed_kernel.thirdparty.triton as triton_topk
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.moe import topk as topk_module
from tokenspeed.runtime.layers.moe.topk import TopKConfig, select_experts


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


@pytest.mark.skipif(
    not current_platform().is_nvidia or not torch.cuda.is_available(),
    reason="MiniMax top-4 Triton routing requires an NVIDIA GPU.",
)
def test_minimax_top4_uses_native_triton_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    hidden_states = torch.randn(23, 64, device="cuda", dtype=torch.bfloat16)
    router_logits = torch.randn(23, 128, device="cuda", dtype=torch.float32)
    correction_bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.05

    def fail_reference(*_args, **_kwargs):
        raise AssertionError("PyTorch routing fallback was used")

    monkeypatch.setattr(
        triton_topk,
        "_biased_grouped_topk_reference",
        fail_reference,
    )
    weights, ids = triton_topk.minimax_biased_grouped_topk(
        hidden_states,
        router_logits,
        correction_bias,
        topk=4,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=2.0,
    )

    scores = router_logits.sigmoid()
    choice_scores = scores + correction_bias
    selected_scores = choice_scores.gather(1, ids.long())
    cutoff = choice_scores.topk(4, dim=-1).values[:, -1:]
    expected_weights = scores.gather(1, ids.long())
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= 2.0

    assert torch.all(selected_scores >= cutoff)
    torch.testing.assert_close(weights, expected_weights)
