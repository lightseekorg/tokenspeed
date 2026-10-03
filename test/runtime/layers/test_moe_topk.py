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


def test_plain_route_uses_kernel_package_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, torch.dtype, bool, float, str | None]] = []

    def fake_topk(
        router_logits: torch.Tensor,
        top_k: int,
        score_function: str,
        selection_method: str,
        renormalize: bool,
        routed_scaling_factor: float,
        topk_indices_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert score_function == "softmax"
        assert selection_method == "topk"
        calls.append(
            (
                top_k,
                topk_indices_dtype,
                renormalize,
                routed_scaling_factor,
                None,
            )
        )
        shape = (router_logits.shape[0], top_k)
        return torch.ones(shape), torch.zeros(shape, dtype=topk_indices_dtype)

    monkeypatch.setattr(topk_module, "moe_topk", fake_topk)
    output = select_experts(
        hidden_states=torch.empty((2, 4), dtype=torch.float32),
        router_logits=torch.empty((2, 8), dtype=torch.float32),
        topk_config=TopKConfig(
            top_k=2,
            router_topk="fused",
            renormalize=True,
            routed_scaling_factor=2.5,
            topk_indices_dtype=torch.int32,
        ),
    )

    assert calls == [(2, torch.int32, True, 2.5, None)]
    assert output.topk_weights.shape == output.topk_ids.shape == (2, 2)
    assert output.topk_ids.dtype == torch.int32


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
            router_topk="fused",
            renormalize=renormalize,
            correction_bias=torch.zeros((8,), dtype=torch.float32),
            routed_scaling_factor=1.0,
        ),
    )

    assert calls == [renormalize]


def _reference_torch_router_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    num_real: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The trainer's eager formula, written out independently of the module."""
    probs = torch.softmax(logits.float(), dim=-1)
    ids = torch.topk(probs + bias, k=top_k, dim=-1, sorted=True).indices
    weights = torch.gather(probs, 1, ids) * scale
    return weights, torch.where(ids >= num_real, torch.full_like(ids, -1), ids)


def test_torch_router_topk_matches_the_reference_formula_with_ties() -> None:
    torch.manual_seed(0)
    num_real, num_zero, top_k, scale = 6, 2, 4, 2.5
    # Quantize so exact ties happen, then bias so a zero expert wins a slot.
    logits = (torch.randn(5, num_real + num_zero) * 2).round() / 2
    logits[0] = 0.0  # a fully tied row
    bias = torch.zeros(num_real + num_zero)
    bias[num_real] = 1.0  # a probability never exceeds 1: this slot wins
    logits = logits.to(torch.bfloat16)

    weights, ids = topk_module.torch_router_topk(
        logits, bias, top_k, num_real, scale, torch.int32
    )
    ref_weights, ref_ids = _reference_torch_router_topk(
        logits, bias, top_k, num_real, scale
    )
    assert ids.dtype == torch.int32
    assert weights.dtype == torch.float32
    assert torch.equal(ids, ref_ids.to(torch.int32))
    assert torch.equal(weights, ref_weights)
    # Zero experts: id -1, weight kept (the unbiased probability, scaled).
    probs = torch.softmax(logits.float(), dim=-1)
    assert (ids[:, 0] == -1).all()
    assert torch.equal(weights[:, 0], probs[:, num_real] * scale)
    # Selected weights are the unbiased probabilities, not the biased scores.
    real = ids[1, 1:].long()
    assert torch.equal(weights[1, 1:], probs[1, real] * scale)
    # Descending biased score within a row (sorted=True).
    biased = probs + bias
    row = torch.where(ids[2] == -1, torch.tensor(num_real), ids[2].long())
    assert (biased[2, row][:-1] >= biased[2, row][1:]).all()


def test_correction_bias_route_can_take_the_torch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_fused(*_args, **_kwargs):
        raise AssertionError("the fused kernel must not run under router_topk=torch")

    monkeypatch.setattr(topk_module, "cuda_routing_flash", refuse_fused)
    logits = torch.randn(3, 8)
    bias = torch.randn(8) * 0.1
    config = TopKConfig(
        top_k=2,
        router_topk="torch",
        renormalize=False,
        correction_bias=bias,
        routed_scaling_factor=1.5,
        zero_expert_num=2,
        topk_indices_dtype=torch.int64,
    )
    output = select_experts(
        hidden_states=torch.empty((3, 4)), router_logits=logits, topk_config=config
    )
    ref_weights, ref_ids = _reference_torch_router_topk(logits, bias, 2, 6, 1.5)
    assert torch.equal(output.topk_ids, ref_ids)
    assert torch.equal(output.topk_weights, ref_weights)
    assert output.topk_ids.dtype == torch.int64


def test_topk_reads_the_router_topk_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "router_topk", "torch")
    assert TopK(top_k=2).topk_config.router_topk == "torch"
    monkeypatch.setitem(global_server_args_dict, "router_topk", "fused")
    assert TopK(top_k=2).topk_config.router_topk == "fused"


def test_torch_router_refuses_renormalization_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    bias = torch.zeros(8)
    monkeypatch.setitem(global_server_args_dict, "router_topk", "torch")
    with pytest.raises(ValueError, match="--router-topk torch"):
        TopK(top_k=2, renormalize=True, correction_bias=bias)
    # Only the correction-bias route is the torch order's; other routes keep
    # their own renormalization, and the fused route always may.
    TopK(top_k=2, renormalize=False, correction_bias=bias)
    TopK(top_k=2, renormalize=True)
    TopK(
        top_k=2,
        renormalize=True,
        correction_bias=bias,
        use_grouped_topk=True,
        num_expert_group=2,
        topk_group=1,
    )
    monkeypatch.setitem(global_server_args_dict, "router_topk", "fused")
    TopK(top_k=2, renormalize=True, correction_bias=bias)
