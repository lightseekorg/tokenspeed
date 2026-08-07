"""Generic TopK's gfx950 E896/top-k16 decode routing integration."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not (torch.cuda.is_available() and current_platform().is_cdna4):
    pytest.skip("AMD CDNA4 is required for Kimi routing tests", allow_module_level=True)


from tokenspeed_kernel.ops.moe import sigmoid_topk as packed_topk_module  # noqa: E402
from tokenspeed_kernel.ops.moe.gluon import sigmoid_topk as routing_module  # noqa: E402
from tokenspeed_kernel.thirdparty.triton import (  # noqa: E402
    minimax_biased_grouped_topk,
)

from tokenspeed.runtime.layers.moe.topk import TopK  # noqa: E402


def _make_topk(correction_bias: torch.Tensor):
    return TopK(
        top_k=16,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=1,
        num_fused_shared_experts=0,
        topk_group=1,
        correction_bias=correction_bias,
        routed_scaling_factor=1.0,
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_generic_topk_uses_k3_decode_or_small_m_gluon_route(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    generator = torch.Generator(device="cuda").manual_seed(20260720 + num_tokens)
    hidden_states = torch.randn(
        (num_tokens, 7168),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    router_logits = torch.randn(
        (num_tokens, 896),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    correction_bias = torch.randn(
        (896,), device="cuda", dtype=torch.float32, generator=generator
    )

    expected_weights, expected_ids = minimax_biased_grouped_topk(
        hidden_states,
        router_logits,
        correction_bias,
        topk=16,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    route_calls = 0
    if num_tokens == 1:
        original_route = packed_topk_module.kimi3_sigmoid_bias_topk
        patch_target = packed_topk_module
        patch_name = "kimi3_sigmoid_bias_topk"
    else:
        original_route = routing_module.invoke_sigmoid_bias_topk_route_gluon
        patch_target = routing_module
        patch_name = "invoke_sigmoid_bias_topk_route_gluon"

    def spy_route(*args, **kwargs):
        nonlocal route_calls
        route_calls += 1
        return original_route(*args, **kwargs)

    monkeypatch.setattr(patch_target, patch_name, spy_route)
    actual = _make_topk(correction_bias)(hidden_states, router_logits)
    torch.cuda.synchronize()
    assert route_calls == 1

    # The reference requests unsorted top-k while the Gluon kernel emits score
    # order. Compare expert/weight pairs in expert-id order.
    expected_order = expected_ids.argsort(dim=-1)
    actual_order = actual.topk_ids.argsort(dim=-1)
    expected_ids = expected_ids.gather(1, expected_order)
    actual_ids = actual.topk_ids.gather(1, actual_order)
    expected_weights = expected_weights.gather(1, expected_order)
    actual_weights = actual.topk_weights.gather(1, actual_order)

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_generic_small_m_topk_is_cuda_graph_capturable(num_tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(20260720 + num_tokens)
    hidden_states = torch.randn(
        (num_tokens, 7168),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    router_logits = torch.randn(
        (num_tokens, 896),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    correction_bias = torch.randn(
        (896,), device="cuda", dtype=torch.float32, generator=generator
    )
    topk = _make_topk(correction_bias)

    eager = topk(hidden_states, router_logits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = topk(hidden_states, router_logits)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured.topk_ids, eager.topk_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        captured.topk_weights, eager.topk_weights, rtol=0, atol=0
    )


def test_generic_topk_uses_prefill_gluon_route(
    monkeypatch: pytest.MonkeyPatch,
):
    generator = torch.Generator(device="cuda").manual_seed(20260725)
    hidden_states = torch.randn(
        (9, 7168), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    router_logits = torch.randn(
        (9, 896), device="cuda", dtype=torch.float32, generator=generator
    )
    correction_bias = torch.randn(
        (896,), device="cuda", dtype=torch.float32, generator=generator
    )

    route_calls = 0
    original_route = routing_module.invoke_sigmoid_bias_topk_route_prefill_gluon

    def spy_route(*args, **kwargs):
        nonlocal route_calls
        route_calls += 1
        return original_route(*args, **kwargs)

    monkeypatch.setattr(
        routing_module,
        "invoke_sigmoid_bias_topk_route_prefill_gluon",
        spy_route,
    )
    actual = _make_topk(correction_bias)(hidden_states, router_logits)
    torch.cuda.synchronize()
    assert route_calls == 1
    expected_weights, expected_ids = minimax_biased_grouped_topk(
        hidden_states,
        router_logits,
        correction_bias,
        topk=16,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )

    actual_order = actual.topk_ids.argsort(dim=-1)
    expected_order = expected_ids.argsort(dim=-1)
    actual_ids = actual.topk_ids.gather(1, actual_order)
    expected_ids = expected_ids.gather(1, expected_order)
    actual_weights = actual.topk_weights.gather(1, actual_order)
    expected_weights = expected_weights.gather(1, expected_order)

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        rtol=5e-3,
        atol=5e-3,
    )
